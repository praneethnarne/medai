import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
torch = pytest.importorskip("torch")

from eval.calibrate import (MALIGNANT, abstention_stats, choose_threshold,  # noqa: E402
                            ece, fit_temperature)


def test_temperature_recovers_overconfidence():
    rng = np.random.default_rng(0)
    true_logits = rng.normal(size=(4000, 5))
    p = torch.softmax(torch.from_numpy(true_logits), 1).numpy()
    y = np.array([rng.choice(5, p=row) for row in p])
    t = fit_temperature(true_logits * 3.0, y)      # model 3x overconfident
    assert 2.5 < t < 3.5


def test_declined_malignant_counts_as_referred():
    # three malignant cases: one predicted malignant, one declined, one missed confidently
    probs = np.array([[0.1, 0.1, 0.8], [0.4, 0.3, 0.3], [0.9, 0.05, 0.05]])
    y = np.array([MALIGNANT] * 3)
    s = abstention_stats(probs, y, threshold=0.5)
    assert s["coverage"] == pytest.approx(2 / 3)
    assert s["malignant_caught_or_referred"] == pytest.approx(2 / 3)
    assert s["malignant_recall_if_always_answering"] == pytest.approx(1 / 3)


def test_threshold_is_lowest_that_meets_target():
    probs = np.array([[0.1, 0.1, 0.8], [0.6, 0.2, 0.2], [0.9, 0.05, 0.05]])
    y = np.array([MALIGNANT, MALIGNANT, 0])
    t, met = choose_threshold(probs, y, target=1.0)
    assert met and t == pytest.approx(0.61)   # just above the missed case's 0.6


def test_ece_zero_when_perfectly_calibrated():
    probs = np.array([[1.0, 0, 0]] * 10)
    assert ece(probs, np.zeros(10, dtype=int))[0] == pytest.approx(0)
