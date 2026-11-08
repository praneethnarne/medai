"""Evaluate a trained checkpoint, with confidence intervals.

    python backend/src/eval/evaluate.py --run ham_baseline_effb0 \
        --collection ham10000 --split test

    # cross-domain: a model trained on dermatoscopic images, judged on phone photos
    python backend/src/eval/evaluate.py --run ham_baseline_effb0 \
        --collection pad_ufes_20 --split test --collapse-to-triage

Point estimates alone are not enough here. PAD-UFES-20's test split holds four
melanoma images, so a per-class recall on it moves 25 points when one image
flips. Every metric is therefore reported with a bootstrap interval and its
support count, and anything resting on fewer than 30 examples is flagged in the
output rather than left for a reader to notice.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import bootstrap  # noqa: F401,E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from data import labels  # noqa: E402
from data.dataset import ISICDataset  # noqa: E402
from train import metrics  # noqa: E402
from train.train import ARCHS, build_model, pick_device  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[3]
LOW_SUPPORT = 30


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    logits_all, y_all = [], []
    for x, y in loader:
        logits = model(x.to(device, non_blocking=True))
        logits_all.append(logits.cpu())
        y_all.extend(y.tolist())
    return torch.cat(logits_all).numpy(), np.asarray(y_all)


def bootstrap_ci(y_true, y_pred, class_names, n_resamples=1000, seed=0, alpha=0.05):
    """Percentile bootstrap over examples.

    Resampling examples -- not predictions -- is what mirrors the sampling
    variation we actually care about: the test set is one draw of patients, and
    a different draw would give a different number.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    keys = ["accuracy", "macro_f1", "macro_recall"]
    draws = {k: [] for k in keys}
    per_class_recall = {c: [] for c in class_names}

    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        rep = metrics.report(y_true[idx], y_pred[idx], class_names)
        for k in keys:
            draws[k].append(rep[k])
        for c in class_names:
            m = rep["per_class"][c]
            per_class_recall[c].append(m["recall"] if m["support"] else np.nan)

    def interval(values):
        arr = np.asarray(values, dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            return None
        lo, hi = np.percentile(arr, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        return [round(float(lo), 4), round(float(hi), 4)]

    return {
        "n_resamples": n_resamples,
        "overall": {k: interval(v) for k, v in draws.items()},
        "per_class_recall": {c: interval(v) for c, v in per_class_recall.items()},
    }


def collapse_logits_to_triage(logits, type_names):
    """Sum probability mass over the lesion types belonging to each triage level."""
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    out = np.zeros((len(probs), len(labels.TRIAGE)))
    for j, t in enumerate(type_names):
        out[:, labels.TRIAGE.index(labels.triage_of_type(t))] += probs[:, j]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run id under results/runs/")
    ap.add_argument("--collection", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--collapse-to-triage", action="store_true",
                    help="collapse a lesion-type model's output to triage, which is "
                         "what makes a model trained on one label set comparable on "
                         "a dataset with a different one")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    run_dir = ROOT / "results" / "runs" / args.run
    config = json.loads((run_dir / "config.json").read_text())
    trained_target = config["target"]
    trained_classes = config["class_names"]

    device = pick_device()
    model, _ = build_model(config["arch"], len(trained_classes))
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu"))
    model = model.to(device)

    eval_target = "triage" if args.collapse_to_triage else trained_target
    if trained_target == "group":
        from data.group_dataset import GroupDataset
        args.collection = "groups"
        ds = GroupDataset(args.split)
    else:
        ds = ISICDataset(args.collection, args.split, target=eval_target)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers)

    logits, y_true = collect_predictions(model, loader, device)

    if args.collapse_to_triage:
        if trained_target != "lesion_type":
            raise SystemExit("--collapse-to-triage needs a lesion_type model")
        scores = collapse_logits_to_triage(logits, trained_classes)
        class_names = list(labels.TRIAGE)
    else:
        scores = logits
        class_names = list(trained_classes)

    y_pred = scores.argmax(1)
    rep = metrics.report(y_true, y_pred, class_names)
    rep["bootstrap_95ci"] = bootstrap_ci(y_true, y_pred, class_names,
                                         n_resamples=args.bootstrap)
    rep["low_support_classes"] = [
        c for c, m in rep["per_class"].items() if 0 < m["support"] < LOW_SUPPORT
    ]
    if trained_target == "group":
        # Shortcut check: spot_growth is almost all PAD-UFES-20. Report SCIN alone, where
        # a "looks like a PAD photo" shortcut cannot help.
        src = np.asarray(ds.sources)
        m = src == "scin"
        rep["scin_only"] = metrics.report(y_true[m], y_pred[m], class_names)
    rep["meta"] = {
        "run": args.run,
        "trained_on": config["collection"],
        "trained_target": trained_target,
        "evaluated_on": args.collection,
        "split": args.split,
        "collapsed_to_triage": args.collapse_to_triage,
        "cross_domain": config["collection"] != args.collection,
        "arch": config["arch"],
    }

    tag = args.tag or f"{args.run}__{args.collection}_{args.split}" + \
        ("__triage" if args.collapse_to_triage else "")
    out = ROOT / "results" / "eval" / f"{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2) + "\n")

    kind = "CROSS-DOMAIN" if rep["meta"]["cross_domain"] else "in-domain"
    print(f"{kind}: trained on {config['collection']}, evaluated on "
          f"{args.collection}/{args.split}\n")
    print(metrics.format_report(rep))
    ci = rep["bootstrap_95ci"]["overall"]
    print(f"\n95% CI  accuracy {ci['accuracy']}  macro F1 {ci['macro_f1']}")
    if "scin_only" in rep:
        print("\nSCIN images only (shortcut check -- spot_growth here is not from PAD-UFES-20):\n")
        print(metrics.format_report(rep["scin_only"]))
    if rep["low_support_classes"]:
        print(f"\nlow support (<{LOW_SUPPORT} examples), treat these as indicative only: "
              f"{', '.join(rep['low_support_classes'])}")
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
