"""Fit an "I don't recognise this image" check for a trained run.

    python backend/src/eval/ood.py --run pad_ft_effb0_phoneaug \
        --fit pad_ufes_20 --check pad_ufes_20 ham10000

Why this exists: the classifier must always pick one of its eight lesion types,
and its confidence only compares those eight with each other. A scar or a rash
can therefore get a confident answer. This check asks a different question
first -- does the image look like anything the model was trained on?

Method (Lee et al., 2018, "A Simple Unified Framework for Detecting
Out-of-Distribution Samples"): take the model's 1,280-number feature vector
just before the final layer, fit one mean per lesion type and one shared
covariance on TRAINING images, and score a new image by its Mahalanobis
distance to the nearest lesion-type mean. Far from every type = unfamiliar.

The threshold is set on VALIDATION images so that --keep (default 95%) of real
lesions pass. What we can measure: how many real lesions are wrongly rejected.
What we cannot measure without a labelled set of non-lesion images: how many
scars, rashes or non-skin photos it catches. Both facts are written out.

Writes results/runs/<run>/ood.npz (means + precision matrix, read by the demo)
and results/runs/<run>/ood.json (threshold and measured rejection rates).
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
from sklearn.covariance import LedoitWolf  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from data.dataset import ISICDataset  # noqa: E402
from train.train import ARCHS, pick_device  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[3]


def feature_layer(model: torch.nn.Module, arch: str) -> torch.nn.Module:
    """The pooling layer whose output is the feature vector fed to the classifier."""
    return model.avgpool


@torch.no_grad()
def extract(model, arch, loader, device) -> tuple[np.ndarray, np.ndarray]:
    feats, ys = [], []
    store = {}
    h = feature_layer(model, arch).register_forward_hook(
        lambda _m, _i, out: store.__setitem__("f", torch.flatten(out, 1)))
    try:
        model.eval()
        for x, y in loader:
            model(x.to(device))
            feats.append(store["f"].cpu().numpy())
            ys.extend(y.tolist())
    finally:
        h.remove()
    return np.concatenate(feats).astype(np.float64), np.asarray(ys)


def fit(feats: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, list[int], np.ndarray]:
    classes = sorted(set(y.tolist()))
    means = np.stack([feats[y == c].mean(0) for c in classes])
    resid = np.concatenate([feats[y == c] - means[i] for i, c in enumerate(classes)])
    precision = LedoitWolf().fit(resid).precision_   # shrinkage: stable with ~1-2k images
    return means, classes, precision


def score(feats: np.ndarray, means: np.ndarray, precision: np.ndarray) -> np.ndarray:
    """Minimum Mahalanobis distance (squared) to any class mean. Higher = less familiar."""
    d = feats[:, None, :] - means[None, :, :]                  # n x k x D
    m = np.einsum("nkd,de,nke->nk", d, precision, d)
    return m.min(1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--fit", nargs="+", default=["pad_ufes_20"],
                    help="collections whose TRAINING images define 'familiar' ('groups' = PAD-UFES-20 + SCIN)")
    ap.add_argument("--check", nargs="+", default=["pad_ufes_20", "ham10000"],
                    help="collections whose TEST images are scored for the report")
    ap.add_argument("--keep", type=float, default=0.95,
                    help="share of real validation lesions that must pass")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    run_dir = ROOT / "results" / "runs" / args.run
    config = json.loads((run_dir / "config.json").read_text())
    arch, names = config["arch"], config["class_names"]
    builder, _ = ARCHS[arch]
    model = builder(weights=None)
    if arch.startswith(("efficientnet", "mobilenet")):
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, len(names))
    else:
        raise SystemExit("ood.py supports efficientnet/mobilenet runs (they expose avgpool)")
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu"))
    device = pick_device()
    model = model.to(device)

    def feats_of(collection, split):
        # eval transforms for every split: we want the features the demo will see
        if collection == "groups":
            from data.group_dataset import GroupDataset
            ds = GroupDataset(split)
            ds.transform = GroupDataset("test").transform
        else:
            ds = ISICDataset(collection, split, target="lesion_type")
            ds.transform = ISICDataset(collection, "test", target="lesion_type").transform
        return extract(model, arch, DataLoader(ds, batch_size=64, num_workers=args.workers), device)

    tr = [feats_of(c, "train") for c in args.fit]
    means, classes, precision = fit(np.concatenate([f for f, _ in tr]),
                                    np.concatenate([y for _, y in tr]))
    val = np.concatenate([score(feats_of(c, "val")[0], means, precision) for c in args.fit])
    threshold = float(np.quantile(val, args.keep))

    report = {"run": args.run, "fitted_on_train_of": args.fit, "keep_target": args.keep,
              "threshold": threshold, "feature_dim": int(means.shape[1]),
              "class_indices": classes,
              "val_rejected": float((val > threshold).mean()),
              "test_rejected": {},
              "not_measured": "share of scars, rashes or non-skin photos caught -- "
                              "needs a labelled set of non-lesion images"}
    for c in args.check:
        s = score(feats_of(c, "test")[0], means, precision)
        report["test_rejected"][c] = float((s > threshold).mean())

    np.savez_compressed(run_dir / "ood.npz", means=means.astype(np.float32),
                        precision=precision.astype(np.float32), threshold=threshold)
    text = json.dumps(report, indent=2) + "\n"
    (run_dir / "ood.json").write_text(text)
    eval_dir = ROOT / "results" / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / f"{args.run}__ood.json").write_text(text)

    print(f"fitted on training images of: {', '.join(args.fit)}")
    print(f"threshold set so {args.keep:.0%} of validation lesions pass "
          f"(actual: {1 - report['val_rejected']:.1%})")
    for c, r in report["test_rejected"].items():
        print(f"  {c:14s} test images flagged as unfamiliar: {r:.1%}")
    print("  scars / rashes / non-skin caught: NOT MEASURED (no labelled non-lesion set)")
    print(f"\nwrote results/runs/{args.run}/ood.npz and ood.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
