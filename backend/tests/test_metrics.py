"""The integrity gate must actually catch an inconsistent report."""

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from train import metrics  # noqa: E402

NAMES = ["a", "b", "c"]


def test_perfect_predictions():
    y = [0, 1, 2, 0, 1, 2]
    rep = metrics.report(y, y, NAMES)
    assert rep["accuracy"] == 1.0
    assert rep["macro_f1"] == 1.0


def test_accuracy_equals_weighted_recall_on_random_data():
    rng = np.random.default_rng(0)
    for _ in range(50):
        y_true = rng.integers(0, 3, 200)
        y_pred = rng.integers(0, 3, 200)
        rep = metrics.report(y_true, y_pred, NAMES)
        assert abs(rep["accuracy"] - rep["weighted_recall"]) < 1e-9


def test_absent_class_does_not_break_macro_average():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    rep = metrics.report(y_true, y_pred, NAMES)  # class "c" never appears
    assert rep["per_class"]["c"]["support"] == 0
    assert 0 < rep["macro_f1"] <= 1


def test_tampered_report_is_rejected():
    y = [0, 1, 2, 0]
    rep = metrics.report(y, y, NAMES)
    rep["accuracy"] = 0.82  # the predecessor repo's fabricated figure
    with pytest.raises(metrics.IntegrityError):
        metrics.assert_integrity(rep)
