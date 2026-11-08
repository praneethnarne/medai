import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from data.labels import LESION_TYPES  # noqa: E402
from demo import care  # noqa: E402


def test_every_lesion_type_has_guidance():
    assert set(care.CONDITIONS) == set(LESION_TYPES)
    for c in care.CONDITIONS.values():
        assert c["urgency"] in care.URGENCY
        assert c["self_care"] and c["red_flags"]


def test_malignant_types_are_never_routine():
    from data.labels import triage_of_type
    for name, c in care.CONDITIONS.items():
        if triage_of_type(name) == "malignant":
            assert c["urgency"] != "routine", name


@pytest.fixture
def fake_run(tmp_path):
    torch = pytest.importorskip("torch")
    from torchvision import models

    classes = list(LESION_TYPES)
    model = models.efficientnet_b0(weights=None)
    model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, len(classes))
    torch.save(model.state_dict(), tmp_path / "best.pt")
    (tmp_path / "config.json").write_text(json.dumps({
        "run_id": "fake", "arch": "efficientnet_b0", "target": "lesion_type",
        "class_names": classes, "collection": "ham10000", "n_train": 0, "n_val": 0,
    }))
    return tmp_path


def test_predict_end_to_end(fake_run):
    import numpy as np
    from PIL import Image

    from demo.inference import load_run, predict

    run = load_run(fake_run)
    img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (300, 400, 3), dtype=np.uint8))
    pred = predict(run, img, threshold=0.5)

    assert pred.heatmap.size == (224, 224)
    assert abs(sum(pred.probs.values()) - 1) < 1e-4
    assert abs(sum(pred.triage_probs.values()) - 1) < 1e-4
    assert set(pred.triage_probs) <= {"benign", "indeterminate", "malignant"}
    assert pred.top_type == next(iter(pred.probs))


def test_abstains_below_threshold(fake_run):
    import numpy as np
    from PIL import Image

    from demo.inference import load_run, predict

    run = load_run(fake_run)
    img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    assert predict(run, img, threshold=1.01).abstained
    assert not predict(run, img, threshold=0.0).abstained


def test_calibration_file_sets_temperature_and_threshold(fake_run):
    from demo.inference import load_run

    (fake_run / "calibration.json").write_text(json.dumps({"temperature": 1.7, "threshold": 0.62}))
    run = load_run(fake_run)
    assert run.temperature == 1.7 and run.threshold == 0.62


def test_phone_capture_keeps_size_and_mode():
    import numpy as np
    from PIL import Image

    from data.dataset import PhoneCapture

    img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (300, 400, 3), dtype=np.uint8))
    aug = PhoneCapture(1, 1, 1, 1)
    out = aug(img)
    assert out.size == img.size and out.mode == "RGB"
