"""Load a trained run, classify one image, and explain it with Grad-CAM.

Kept free of any UI code so it can be tested and reused by the API later.

    model = load_run("results/runs/ham_effb0_colab")
    result = predict(model, pil_image)
"""

from __future__ import annotations

import json
import pathlib
import sys
from dataclasses import dataclass

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from data.dataset import build_transforms  # noqa: E402
from data.labels import DISPLAY_NAMES, triage_of_type  # noqa: E402
from train.train import ARCHS  # noqa: E402


@dataclass
class LoadedRun:
    model: torch.nn.Module
    class_names: list[str]
    arch: str
    run_id: str
    device: torch.device
    target_layer: torch.nn.Module
    temperature: float = 1.0          # from calibration.json, 1.0 if uncalibrated
    threshold: float | None = None    # calibrated abstention threshold, if any
    ood: dict | None = None           # from ood.npz: means, precision, threshold
    target: str = "lesion_type"       # "lesion_type" (Level 2 spots) or "group" (Level 1)


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _target_layer(model: torch.nn.Module, arch: str) -> torch.nn.Module:
    """The last convolutional block: highest-level features that still have a
    spatial layout, which is what Grad-CAM needs."""
    if arch.startswith(("efficientnet", "mobilenet")):
        return model.features[-1]
    return model.layer4[-1]  # resnet


def load_run(run_dir: str | pathlib.Path, device: torch.device | None = None) -> LoadedRun:
    run_dir = pathlib.Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    if config["target"] not in ("lesion_type", "group"):
        raise ValueError("the demo expects a lesion_type or group run")
    arch = config["arch"]
    class_names = config["class_names"]

    builder, _ = ARCHS[arch]
    model = builder(weights=None)  # weights come from best.pt, no download needed
    if arch.startswith(("efficientnet", "mobilenet")):
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features,
                                               len(class_names))
    else:
        model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu"))

    device = device or _pick_device()
    model = model.to(device).eval()
    temperature, threshold = 1.0, None
    cal = run_dir / "calibration.json"
    if cal.exists():
        c = json.loads(cal.read_text())
        temperature, threshold = float(c["temperature"]), float(c["threshold"])
    ood = None
    if (run_dir / "ood.npz").exists():
        z = np.load(run_dir / "ood.npz")
        ood = {"means": z["means"].astype(np.float64),
               "precision": z["precision"].astype(np.float64),
               "threshold": float(z["threshold"])}
    return LoadedRun(model, class_names, arch, config["run_id"], device,
                     _target_layer(model, arch), temperature, threshold, ood, config["target"])


_EVAL_TF = build_transforms(train=False)


def preprocess(image: Image.Image) -> tuple[torch.Tensor, Image.Image]:
    """Returns the model input and the exact 224x224 crop the model saw,
    so the heatmap can be overlaid on the right pixels."""
    image = image.convert("RGB")
    x = _EVAL_TF(image).unsqueeze(0)
    # Undo normalisation to recover the cropped view for display.
    from data.dataset import MEAN, STD
    view = x[0].clone()
    for c in range(3):
        view[c] = view[c] * STD[c] + MEAN[c]
    view = (view.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return x, Image.fromarray(view)


def grad_cam(run: LoadedRun, x: torch.Tensor, class_idx: int) -> np.ndarray:
    """Standard Grad-CAM (Selvaraju et al., 2017): weight each channel of the
    target layer by the mean gradient of the class score, sum, ReLU, normalise.
    Returns an HxW array in [0, 1] at the input resolution."""
    store: dict[str, torch.Tensor] = {}

    def fwd_hook(_m, _i, out):
        store["act"] = out
        out.register_hook(lambda g: store.__setitem__("grad", g))

    handle = run.target_layer.register_forward_hook(fwd_hook)
    try:
        x = x.to(run.device).requires_grad_(False)
        with torch.enable_grad():
            logits = run.model(x)
            run.model.zero_grad(set_to_none=True)
            logits[0, class_idx].backward()
    finally:
        handle.remove()

    act, grad = store["act"][0], store["grad"][0]           # C x h x w
    weights = grad.mean(dim=(1, 2))                        # C
    cam = torch.relu((weights[:, None, None] * act).sum(0))
    cam = torch.nn.functional.interpolate(cam[None, None], size=x.shape[-2:],
                                          mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam.detach().cpu().numpy()


def overlay(view: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Blend a blue-to-red heatmap over the image (no matplotlib dependency)."""
    c = np.clip(cam, 0, 1)
    heat = np.stack([
        np.clip(1.5 - np.abs(4 * c - 3), 0, 1),   # red   peaks at high values
        np.clip(1.5 - np.abs(4 * c - 2), 0, 1),   # green peaks in the middle
        np.clip(1.5 - np.abs(4 * c - 1), 0, 1),   # blue  peaks at low values
    ], axis=-1)
    base = np.asarray(view, dtype=np.float32) / 255.0
    out = (1 - alpha) * base + alpha * heat
    return Image.fromarray((out * 255).astype(np.uint8))


@dataclass
class Prediction:
    probs: dict[str, float]        # lesion type -> probability, sorted high to low
    top_type: str
    top_prob: float
    triage_probs: dict[str, float]
    abstained: bool
    heatmap: Image.Image
    view: Image.Image
    unfamiliar: bool = False          # failed the "does this look like a lesion I know" check
    ood_score: float | None = None


def _forward(run: LoadedRun, x: torch.Tensor) -> tuple[np.ndarray, bool, float | None]:
    """Calibrated probabilities, plus the familiarity check on the same forward pass."""
    store: dict = {}
    hook = run.model.avgpool.register_forward_hook(
        lambda _m, _i, out: store.__setitem__("f", torch.flatten(out, 1)))
    try:
        with torch.no_grad():
            logits = run.model(x.to(run.device)) / run.temperature
            p = torch.softmax(logits, dim=1)[0].cpu().numpy()
    finally:
        hook.remove()

    unfamiliar, ood_score = False, None
    if run.ood is not None:
        f = store["f"][0].cpu().numpy().astype(np.float64)
        d = f[None, :] - run.ood["means"]
        ood_score = float(np.einsum("kd,de,ke->k", d, run.ood["precision"], d).min())
        unfamiliar = ood_score > run.ood["threshold"]
    return p, unfamiliar, ood_score


def predict(run: LoadedRun, image: Image.Image, threshold: float) -> Prediction:
    x, view = preprocess(image)
    p, unfamiliar, ood_score = _forward(run, x)

    order = np.argsort(-p)
    probs = {run.class_names[i]: float(p[i]) for i in order}
    top = int(order[0])

    triage: dict[str, float] = {}
    for name, prob in probs.items():
        level = triage_of_type(name)
        triage[level] = triage.get(level, 0.0) + prob

    cam = grad_cam(run, x, top)
    # Abstain on the triage level -- the answer the user acts on -- matching
    # how calibrate.py chose the threshold.
    top_triage = max(triage.values())
    return Prediction(
        probs=probs,
        top_type=run.class_names[top],
        top_prob=float(p[top]),
        triage_probs=dict(sorted(triage.items(), key=lambda kv: -kv[1])),
        abstained=unfamiliar or top_triage < threshold,
        heatmap=overlay(view, cam),
        view=view,
        unfamiliar=unfamiliar,
        ood_score=ood_score,
    )


@dataclass
class GroupPrediction:
    probs: dict[str, float]        # group -> probability, sorted high to low
    top_group: str
    top_prob: float
    abstained: bool
    unfamiliar: bool
    heatmap: Image.Image
    ood_score: float | None = None


def predict_group(run: LoadedRun, image: Image.Image, threshold: float) -> GroupPrediction:
    """Level 1: what kind of thing is on the skin."""
    x, view = preprocess(image)
    p, unfamiliar, ood_score = _forward(run, x)
    order = np.argsort(-p)
    top = int(order[0])
    return GroupPrediction(
        probs={run.class_names[i]: float(p[i]) for i in order},
        top_group=run.class_names[top],
        top_prob=float(p[top]),
        abstained=unfamiliar or float(p[top]) < threshold,
        unfamiliar=unfamiliar,
        heatmap=overlay(view, grad_cam(run, x, top)),
        ood_score=ood_score,
    )


def display_name(lesion_type: str) -> str:
    return DISPLAY_NAMES.get(lesion_type, lesion_type)
