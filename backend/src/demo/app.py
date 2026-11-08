"""MEdAI demo: upload a skin photo, get a prediction, a heatmap and care guidance.

    python backend/src/demo/app.py --run ham_effb0_colab            # local
    python backend/src/demo/app.py --run ham_effb0_colab --share    # public link (Colab)

Every number shown in the "About this model" panel is read from the run's own
files under results/runs/<run>/, never typed by hand.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import bootstrap  # noqa: F401,E402

import gradio as gr  # noqa: E402

from demo import care  # noqa: E402
from demo.inference import display_name, load_run, predict  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[3]

# Used only when the run has no calibration.json. A calibrated run supplies its
# own threshold, chosen on validation data by eval/calibrate.py.
UNCALIBRATED_THRESHOLD = 0.5

SCOPE_NOTE = (
    "MEdAI only recognises eight kinds of skin lesion: mole, benign keratosis, "
    "actinic keratosis, basal cell carcinoma, squamous cell carcinoma, melanoma, "
    "vascular lesion and dermatofibroma. Scars, wounds, rashes, insect bites and "
    "other skin conditions are outside what it can judge -- its answer on them "
    "means nothing."
)


def _model_facts(run_dir: pathlib.Path) -> str:
    """Summarise what this model is, using only measured, saved values."""
    config = json.loads((run_dir / "config.json").read_text())
    lines = [f"**Model:** {config['arch']}, run `{config['run_id']}`",
             f"**Trained on:** {config['collection']} "
             f"({config['n_train']} training images, {config['n_val']} validation)"]

    best = run_dir / "best_val_report.json"
    if best.exists():
        b = json.loads(best.read_text())
        lines.append(f"**Best validation macro F1:** {b['macro_f1']:.3f} "
                     f"(stage {b['stage']}, epoch {b['epoch']})")
    else:
        lines.append("**Validation metrics:** not available yet (training not finished)")

    cal = run_dir / "calibration.json"
    if cal.exists():
        c = json.loads(cal.read_text())
        t = c["test"]
        pct = lambda v: "n/a" if v is None else f"{v:.0%}"  # noqa: E731
        lines.append(
            f"**Calibrated** on {c['collection']} validation data: temperature "
            f"{c['temperature']:.2f}, confidence threshold {c['threshold']:.2f}. "
            f"On the {c['collection']} test set it answered {pct(t['coverage'])} of "
            f"images, and {pct(t['malignant_caught_or_referred'])} of malignant cases "
            f"were either flagged malignant or referred."
        )
    else:
        lines.append("**Not calibrated:** confidence scores are raw model output.")

    lines.append(
        f"\nTraining images: **{'smartphone photos' if config['collection'] == 'pad_ufes_20' else 'dermatoscope images'}**"
        + ("" if config.get("init_from") is None else " (starting from a model pretrained on dermatoscope images)")
        + "."
    )
    return "\n\n".join(lines)


def _guidance_md(pred, threshold: float) -> str:
    if pred.abstained:
        u = care.UNSURE
        return (f"## {u['label']}\n\n{u['note']}\n\n"
                f"_Top triage score was below the {threshold:.0%} confidence "
                f"threshold. Top lesion-type guess: {display_name(pred.top_type)} "
                f"({pred.top_prob:.0%})._\n\n_{SCOPE_NOTE}_\n\n---\n_{care.DISCLAIMER}_")

    c = care.CONDITIONS[pred.top_type]
    u = care.URGENCY[c["urgency"]]
    parts = [
        f"## {u['label']}",
        u["note"],
        f"### Most likely: {display_name(pred.top_type)} ({pred.top_prob:.0%})",
        f"_{c['plain']}_",
        f"**What it is.** {c['what_it_is']}",
        f"**When to get it checked.** {c['seek_care']}",
        "**While you wait**\n" + "\n".join(f"- {s}" for s in c["self_care"]),
        "**See a doctor sooner if**\n" + "\n".join(f"- {s}" for s in c["red_flags"]),
        "---",
        f"_{care.DISCLAIMER} This result comes from a model that can be wrong._",
    ]
    return "\n\n".join(parts)


def build_app(run_dir: pathlib.Path, threshold: float | None) -> gr.Blocks:
    run = load_run(run_dir)
    calibrated = run.threshold is not None
    if threshold is None:
        threshold = run.threshold if calibrated else UNCALIBRATED_THRESHOLD

    def analyse(image, thr):
        if image is None:
            return None, None, None, "Upload a photo to begin."
        pred = predict(run, image, thr)
        types = {display_name(k): v for k, v in pred.probs.items()}
        triage = {k.capitalize(): v for k, v in pred.triage_probs.items()}
        if pred.abstained:
            # Never headline a verdict the guidance below has just withheld:
            # a large "Benign" over "please see a doctor" reads as reassurance.
            return (pred.heatmap,
                    gr.Label(value="Not confident -- see a doctor",
                             label="Triage"),
                    gr.Label(value=types,
                             label="Lesion type (top guesses, below the confidence threshold)"),
                    _guidance_md(pred, thr))
        return (pred.heatmap,
                gr.Label(value=triage, label="Triage"),
                gr.Label(value=types, label="Lesion type"),
                _guidance_md(pred, thr))

    with gr.Blocks(title="MEdAI") as app:
        gr.Markdown("# MEdAI\nSkin-lesion screening aid. "
                    "**A first check, not a diagnosis.**\n\n" + SCOPE_NOTE)
        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(type="pil", label="Photo of the skin spot")
                thr = gr.Slider(0.34, 0.99, value=threshold, step=0.01,
                                label=("Confidence needed to answer "
                                       + (f"(calibrated default {run.threshold:.2f})"
                                          if calibrated else "(uncalibrated placeholder)")))
                btn = gr.Button("Analyse", variant="primary")
                with gr.Accordion("About this model", open=False):
                    gr.Markdown(_model_facts(run_dir))
            with gr.Column(scale=1):
                heat = gr.Image(label="Where the model looked (Grad-CAM; red = most influence)")
                triage = gr.Label(label="Triage", num_top_classes=3)
                types = gr.Label(label="Lesion type", num_top_classes=4)
        guidance = gr.Markdown()

        btn.click(analyse, [image, thr], [heat, triage, types, guidance])
        image.upload(analyse, [image, thr], [heat, triage, types, guidance])
    return app


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run id under results/runs/, or a path")
    ap.add_argument("--threshold", type=float, default=None,
                    help="override the run's calibrated threshold")
    ap.add_argument("--share", action="store_true", help="create a public gradio.live link")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    run_dir = pathlib.Path(args.run)
    if not run_dir.is_dir():
        run_dir = ROOT / "results" / "runs" / args.run
    if not (run_dir / "best.pt").exists():
        print(f"no best.pt in {run_dir} -- has training saved a checkpoint yet?")
        return 1

    build_app(run_dir, args.threshold).launch(share=args.share, server_port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
