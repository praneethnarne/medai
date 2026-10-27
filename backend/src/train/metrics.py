"""Classification metrics, with the integrity gate the project ground rules require.

Accuracy alone is misleading on these datasets, so every report carries per-class
precision, recall and F1 alongside it, with support counts so nobody mistakes a
metric computed on four examples for a reliable one.
"""

from __future__ import annotations

import numpy as np


def confusion(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def report(y_true, y_pred, class_names) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    k = len(class_names)
    cm = confusion(y_true, y_pred, k)

    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    tp = np.diag(cm)

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted > 0, tp / np.maximum(predicted, 1), 0.0)
        recall = np.where(support > 0, tp / np.maximum(support, 1), 0.0)
        denom = precision + recall
        f1 = np.where(denom > 0, 2 * precision * recall / np.maximum(denom, 1e-12), 0.0)

    present = support > 0
    total = support.sum()
    accuracy = float(tp.sum() / total) if total else 0.0

    out = {
        "accuracy": accuracy,
        "macro_precision": float(precision[present].mean()) if present.any() else 0.0,
        "macro_recall": float(recall[present].mean()) if present.any() else 0.0,
        "macro_f1": float(f1[present].mean()) if present.any() else 0.0,
        "weighted_recall": float((recall * support).sum() / total) if total else 0.0,
        "weighted_f1": float((f1 * support).sum() / total) if total else 0.0,
        "n": int(total),
        "per_class": {
            name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, name in enumerate(class_names)
        },
        "confusion_matrix": cm.tolist(),
        "class_names": list(class_names),
    }
    assert_integrity(out)
    return out


class IntegrityError(AssertionError):
    """Raised when a report is internally inconsistent."""


def assert_integrity(rep: dict, tol: float = 1e-6) -> None:
    """Accuracy must equal weighted-average recall.

    This identity holds for any confusion matrix. The predecessor codebase this
    project replaced shipped a report where it did not -- accuracy read 0.82
    while weighted recall read 0.72, because the number had been edited by hand.
    Checking it here means a fabricated or mismatched metric cannot pass
    silently through the pipeline.
    """
    diff = abs(rep["accuracy"] - rep["weighted_recall"])
    if diff > tol:
        raise IntegrityError(
            f"accuracy {rep['accuracy']:.6f} != weighted recall "
            f"{rep['weighted_recall']:.6f} (diff {diff:.2e}); the report is "
            "internally inconsistent and must not be published"
        )


def format_report(rep: dict) -> str:
    lines = [
        f"{'class':24s} {'prec':>6s} {'recall':>7s} {'f1':>6s} {'support':>8s}",
        "-" * 55,
    ]
    for name, m in rep["per_class"].items():
        flag = "  <- low support" if 0 < m["support"] < 30 else ""
        lines.append(f"{name:24s} {m['precision']:6.3f} {m['recall']:7.3f} "
                     f"{m['f1']:6.3f} {m['support']:8d}{flag}")
    lines += [
        "-" * 55,
        f"{'accuracy':24s} {rep['accuracy']:20.3f} {rep['n']:8d}",
        f"{'macro f1':24s} {rep['macro_f1']:20.3f}",
        f"{'weighted f1':24s} {rep['weighted_f1']:20.3f}",
    ]
    return "\n".join(lines)
