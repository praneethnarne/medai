"""Calibrate a run's confidence and choose its abstention threshold from data.

    python backend/src/eval/calibrate.py --run pad_ft_effb0 --collection pad_ufes_20

Two steps, both fitted on the VALIDATION split and then reported on TEST:

1. Temperature scaling (Guo et al., 2017). One number T divides the logits so
   that a predicted 80% is right about 80% of the time. It never changes which
   class wins, only how sure the model claims to be. Measured with expected
   calibration error (ECE) before and after.

2. Abstention threshold. The demo declines to answer when the top triage
   probability is below a threshold, and a declined case is sent to a doctor.
   So for safety, a malignant case counts as caught if it is either predicted
   malignant or declined. We pick the LOWEST threshold (the one that answers
   most often) whose caught-malignant rate on validation reaches --target-recall.
   Melanoma alone is too rare in PAD-UFES-20 (a handful of test images) to set
   a threshold on, so the target is on the malignant triage level as a whole.

Writes results/runs/<run>/calibration.json, which the demo reads, and a copy
under results/eval/.
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
from eval.evaluate import collect_predictions  # noqa: E402
from train.train import ARCHS, pick_device  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[3]
MALIGNANT = labels.TRIAGE.index("malignant")


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Minimise validation negative log-likelihood over log T (keeps T > 0)."""
    lg = torch.from_numpy(logits).float()
    yt = torch.from_numpy(y).long()
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(lg / log_t.exp(), yt)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def triage_probs(logits: np.ndarray, type_names: list[str], temperature: float) -> np.ndarray:
    p = torch.softmax(torch.from_numpy(logits).float() / temperature, dim=1).numpy()
    out = np.zeros((len(p), len(labels.TRIAGE)))
    for j, t in enumerate(type_names):
        out[:, labels.TRIAGE.index(labels.triage_of_type(t))] += p[:, j]
    return out


def ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> tuple[float, list[dict]]:
    """Expected calibration error on the top-class probability, with the bins
    (for a reliability diagram)."""
    conf = probs.max(1)
    correct = (probs.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    total, bins = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if not m.any():
            continue
        gap = abs(conf[m].mean() - correct[m].mean())
        total += m.mean() * gap
        bins.append({"lo": round(float(lo), 2), "hi": round(float(hi), 2), "n": int(m.sum()),
                     "mean_confidence": round(float(conf[m].mean()), 4),
                     "accuracy": round(float(correct[m].mean()), 4)})
    return float(total), bins


def abstention_stats(probs: np.ndarray, y: np.ndarray, threshold: float) -> dict:
    answered = probs.max(1) >= threshold
    pred = probs.argmax(1)
    mal = y == MALIGNANT
    caught = mal & (~answered | (pred == MALIGNANT))
    return {
        "coverage": float(answered.mean()),
        "accuracy_on_answered": float((pred[answered] == y[answered]).mean()) if answered.any() else None,
        "malignant_caught_or_referred": float(caught[mal].mean()) if mal.any() else None,
        "malignant_recall_if_always_answering": float((pred[mal] == MALIGNANT).mean()) if mal.any() else None,
        "n": int(len(y)),
        "n_malignant": int(mal.sum()),
    }


def choose_threshold(probs: np.ndarray, y: np.ndarray, target: float) -> tuple[float, bool]:
    for t in np.round(np.arange(0.34, 1.0, 0.01), 2):  # below 1/3 nothing is ever declined
        s = abstention_stats(probs, y, float(t))
        if s["malignant_caught_or_referred"] is not None and s["malignant_caught_or_referred"] >= target:
            return float(t), True
    return 0.99, False


def bootstrap(probs, y, threshold, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    keys = ["coverage", "malignant_caught_or_referred", "accuracy_on_answered"]
    draws = {k: [] for k in keys}
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        s = abstention_stats(probs[idx], y[idx], threshold)
        for k in keys:
            if s[k] is not None:
                draws[k].append(s[k])
    return {k: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
            for k, v in draws.items() if v}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--collection", required=True)
    ap.add_argument("--target-recall", type=float, default=0.90,
                    help="share of malignant cases that must be flagged or referred (validation)")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    run_dir = ROOT / "results" / "runs" / args.run
    config = json.loads((run_dir / "config.json").read_text())
    type_names = config["class_names"]
    builder, _ = ARCHS[config["arch"]]
    model = builder(weights=None)
    if config["arch"].startswith(("efficientnet", "mobilenet")):
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, len(type_names))
    else:
        model.fc = torch.nn.Linear(model.fc.in_features, len(type_names))
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu"))
    device = pick_device()
    model = model.to(device).eval()

    def logits_for(split):
        ds = ISICDataset(args.collection, split, target="lesion_type")
        lg, y_type = collect_predictions(model, DataLoader(ds, batch_size=64, num_workers=args.workers), device)
        y_tri = np.array([labels.TRIAGE.index(labels.triage_of_type(type_names[i])) for i in y_type])
        return lg, y_type, y_tri

    val_lg, val_type, val_tri = logits_for("val")
    test_lg, _, test_tri = logits_for("test")

    T = fit_temperature(val_lg, val_type)
    out = {"run": args.run, "collection": args.collection, "temperature": round(T, 4),
           "fitted_on": "val", "target_malignant_caught_or_referred": args.target_recall}

    for name, temp in (("before", 1.0), ("after", T)):
        for split, lg, y in (("val", val_lg, val_tri), ("test", test_lg, test_tri)):
            e, bins = ece(triage_probs(lg, type_names, temp), y)
            out.setdefault("ece_triage", {}).setdefault(name, {})[split] = round(e, 4)
            if split == "test":
                out.setdefault("reliability_test", {})[name] = bins

    val_p = triage_probs(val_lg, type_names, T)
    test_p = triage_probs(test_lg, type_names, T)
    thr, met = choose_threshold(val_p, val_tri, args.target_recall)
    out["threshold"] = thr
    out["threshold_meets_target_on_val"] = met
    out["val"] = abstention_stats(val_p, val_tri, thr)
    out["test"] = abstention_stats(test_p, test_tri, thr)
    out["test_ci95"] = bootstrap(test_p, test_tri, thr)

    text = json.dumps(out, indent=2) + "\n"
    (run_dir / "calibration.json").write_text(text)
    eval_dir = ROOT / "results" / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / f"{args.run}__{args.collection}_calibration.json").write_text(text)

    e = out["ece_triage"]
    t = {k: ("n/a" if v is None else (f"{v:.3f}" if isinstance(v, float) else v))
         for k, v in out["test"].items()}
    ci = out["test_ci95"]
    print(f"temperature T = {T:.3f}  (fitted on {args.collection}/val)")
    print(f"ECE (triage, test): {e['before']['test']:.3f} before -> {e['after']['test']:.3f} after")
    print(f"threshold = {thr:.2f}  chosen on val for >= {args.target_recall:.0%} malignant "
          f"caught-or-referred{'' if met else '  (TARGET NOT MET on val -- using 0.99)'}")
    print(f"TEST  n={t['n']}  malignant={t['n_malignant']}")
    print(f"  answered (coverage)            {t['coverage']}  CI {ci.get('coverage')}")
    print(f"  accuracy on answered           {t['accuracy_on_answered']}  CI {ci.get('accuracy_on_answered')}")
    print(f"  malignant caught or referred   {t['malignant_caught_or_referred']}  CI {ci.get('malignant_caught_or_referred')}")
    print(f"  malignant recall, never declining  {t['malignant_recall_if_always_answering']}")
    print(f"\nwrote results/runs/{args.run}/calibration.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
