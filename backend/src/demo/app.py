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

# Placeholder, NOT a calibrated value. Calibration and a threshold chosen
# against a target melanoma recall are roadmap item 5. Until then the demo says
# so on screen.
DEFAULT_THRESHOLD = 0.5


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

    for path in sorted(run_dir.glob("eval_*.json")):
        lines.append(f"**Evaluation file:** `{path.name}`")

    lines.append(
        "\nTrained on **dermatoscopic** images. Phone photos are a different "
        "domain, so expect lower accuracy on them until the PAD-UFES-20 "
        "fine-tune (roadmap item 3) is done."
    )
    return "\n\n".join(lines)


def _guidance_md(pred, threshold: float) -> str:
    if pred.abstained:
        u = care.UNSURE
        return (f"## {u['label']}\n\n{u['note']}\n\n"
                f"_Top guess was {display_name(pred.top_type)} at "
                f"{pred.top_prob:.0%}, below the {threshold:.0%} confidence "
                f"threshold._\n\n---\n_{care.DISCLAIMER}_")

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


def build_app(run_dir: pathlib.Path, threshold: float) -> gr.Blocks:
    run = load_run(run_dir)

    def analyse(image, thr):
        if image is None:
            return None, None, None, "Upload a photo to begin."
        pred = predict(run, image, thr)
        types = {display_name(k): v for k, v in pred.probs.items()}
        triage = {k.capitalize(): v for k, v in pred.triage_probs.items()}
        return pred.heatmap, triage, types, _guidance_md(pred, thr)

    with gr.Blocks(title="MEdAI") as app:
        gr.Markdown("# MEdAI\nSkin-lesion screening aid. "
                    "**A first check, not a diagnosis.**")
        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(type="pil", label="Photo of the skin spot")
                thr = gr.Slider(0.2, 0.9, value=threshold, step=0.05,
                                label="Confidence needed to answer "
                                      "(uncalibrated placeholder)")
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
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
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
