import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
pytest.importorskip("torch")
pytest.importorskip("sklearn")

from eval.ood import fit, score  # noqa: E402


def test_far_points_score_higher_than_training_like_points():
    rng = np.random.default_rng(0)
    centres = rng.normal(scale=5, size=(3, 20))
    y = rng.integers(0, 3, 900)
    feats = centres[y] + rng.normal(size=(900, 20))
    means, classes, precision = fit(feats, y)
    near = centres[[0, 1, 2]] + rng.normal(size=(3, 20))
    far = rng.normal(loc=40, size=(3, 20))
    assert score(far, means, precision).min() > score(near, means, precision).max()
    assert classes == [0, 1, 2]
