"""JSON prediction for other front ends (the React interface in medai-mockup).

Exposed by app.py as the Gradio API endpoint `/predict_json`. One call returns
everything a result page needs, with images as data URLs so the caller needs no
second request:

    status        "answer" | "unsure" | "unfamiliar"
    kind          Level 1 result when a group model is loaded, else null
    predicted     lesion type as the mockup's short code (mel, bcc, scc, akiec, bkl, nv, df, vasc)
    confidence    probability of that type (calibrated if the run is)
    triage        {benign, indeterminate, malignant} probabilities
    probabilities {code: probability} for all eight types
    cropUrl       the exact 224x224 square the model saw
    heatUrl       Grad-CAM heatmap alone, transparent PNG, aligned with cropUrl
    threshold     confidence needed to answer; calibrated says where it came from
"""

from __future__ import annotations

import base64
import io
import time
import uuid

from PIL import Image

from demo.inference import LoadedRun, predict, predict_group, preprocess

# This repo's lesion types -> the mockup's condition codes (HAM10000 style, plus scc,
# which ISIC's taxonomy separates from actinic keratosis).
CODE = {
    "nevus": "nv",
    "benign_keratosis": "bkl",
    "actinic_keratosis": "akiec",
    "basal_cell_carcinoma": "bcc",
    "squamous_cell_carcinoma": "scc",
    "melanoma": "mel",
    "vascular": "vasc",
    "dermatofibroma": "df",
}


def _data_url(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt, **({"quality": 90} if fmt == "JPEG" else {}))
    mime = "image/png" if fmt == "PNG" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(buf.getvalue()).decode()


def predict_json(run: LoadedRun, grun: LoadedRun | None, image: Image.Image,
                 threshold: float, group_threshold: float) -> dict:
    started = time.perf_counter()
    image = image.convert("RGB")
    out: dict = {
        "id": uuid.uuid4().hex[:8],
        "simulated": False,
        "model": run.run_id,
        "calibrated": run.threshold is not None,
        "threshold": threshold,
        "kind": None,
    }

    if grun is not None:
        gp = predict_group(grun, image, group_threshold)
        out["kind"] = {"top": gp.top_group, "confidence": gp.top_prob,
                       "probabilities": gp.probs, "abstained": gp.abstained,
                       "model": grun.run_id}
        if gp.unfamiliar or gp.abstained or gp.top_group != "spot_growth":
            status = "unfamiliar" if gp.unfamiliar else ("unsure" if gp.abstained else "answer")
            out.update(status=status, predicted=None, confidence=gp.top_prob,
                       triage=None, probabilities=None,
                       cropUrl=_data_url(gp.view, "JPEG"), heatUrl=_data_url(gp.heat))
            out["inferenceMs"] = round((time.perf_counter() - started) * 1000)
            return out

    pred = predict(run, image, threshold)
    out.update(
        status="unfamiliar" if pred.unfamiliar else ("unsure" if pred.abstained else "answer"),
        predicted=CODE[pred.top_type],
        confidence=pred.top_prob,
        triage=pred.triage_probs,
        probabilities={CODE[k]: v for k, v in pred.probs.items()},
        cropUrl=_data_url(pred.view, "JPEG"),
        heatUrl=_data_url(pred.heat),
    )
    out["inferenceMs"] = round((time.perf_counter() - started) * 1000)
    return out


__all__ = ["predict_json", "CODE", "preprocess"]
