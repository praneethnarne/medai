"""MEdAI demo: upload a skin photo, get a prediction, a heatmap and care guidance.

    python backend/src/demo/app.py --run ham_effb0_colab            # local
    python backend/src/demo/app.py --run ham_effb0_colab --share    # public link (Colab)

    # two levels (ADR 0007): first what kind of thing, then which spot or growth
    python backend/src/demo/app.py --groups-run groups_effb0 --run pad_ft_effb0_phoneaug --share

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
from data.groups import DISPLAY as GROUP_DISPLAY  # noqa: E402
from demo.api import predict_json  # noqa: E402
from demo.inference import display_name, load_run, predict, predict_group  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[3]

# Used only when the run has no calibration.json. A calibrated run supplies its
# own threshold, chosen on validation data by eval/calibrate.py.
UNCALIBRATED_THRESHOLD = 0.5

SCOPE_NOTE = (
    "This model checks skin spots and growths -- moles, keratoses and the kinds that can "
    "be skin cancer (eight types in all). Scars, wounds, rashes, insect bites and other "
    "skin conditions are outside what it can judge -- its answer on them means nothing."
)

SCOPE_NOTE_GROUPS = (
    "MEdAI first works out what kind of thing is on the skin -- a spot or growth, a rash, "
    "an infection, a bite or sting, or a wound, burn, bruise or scar -- and then, for spots "
    "and growths, which kind. Anything else, or a photo it does not recognise, gets no answer."
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

    ood = run_dir / "ood.json"
    if ood.exists():
        o = json.loads(ood.read_text())
        rej = ", ".join(f"{k} {v:.0%}" for k, v in o["test_rejected"].items())
        lines.append(
            f"**Unfamiliar-image check:** on. Set so {1 - o['val_rejected']:.0%} of real "
            f"validation lesions pass. Test images flagged as unfamiliar: {rej}. "
            f"How many scars, rashes or non-skin photos it catches has not been measured."
        )
    else:
        lines.append("**Unfamiliar-image check:** off (no ood.npz for this run).")

    lines.append(
        f"\nTraining images: **{'smartphone photos' if config['collection'] == 'pad_ufes_20' else 'dermatoscope images'}**"
        + ("" if config.get("init_from") is None else " (starting from a model pretrained on dermatoscope images)")
        + "."
    )
    return "\n\n".join(lines)


NOT_APPLICABLE = "_Scar, wound, rash or bite? This result does not apply -- see the note at the top._"


def _guidance_md(pred, threshold: float) -> str:
    if pred.unfamiliar:
        u = care.UNFAMILIAR
        return (f"## {u['label']}\n\n{u['note']}\n\n_{SCOPE_NOTE}_"
                f"\n\n---\n_{care.DISCLAIMER}_")
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
        NOT_APPLICABLE,
        f"### If this is a skin lesion, it looks most like: {display_name(pred.top_type)} ({pred.top_prob:.0%})",
        f"_{c['plain']}_",
        f"**What it is.** {c['what_it_is']}",
        f"**When to get it checked.** {c['seek_care']}",
        "**While you wait**\n" + "\n".join(f"- {s}" for s in c["self_care"]),
        "**See a doctor sooner if**\n" + "\n".join(f"- {s}" for s in c["red_flags"]),
        "---",
        f"_{care.DISCLAIMER} This result comes from a model that can be wrong._",
    ]
    return "\n\n".join(parts)


def _group_md(gp) -> str:
    g = care.GROUP_GUIDANCE[gp.top_group]
    e = care.EMERGENCY
    parts = [
        f"## {g['headline']}",
        f"**{e['label']}:**\n" + "\n".join(f"- {x}" for x in e["signs"]),
        f"### It looks most like: {GROUP_DISPLAY[gp.top_group]} ({gp.top_prob:.0%})",
        f"_{g['plain']}_",
        f"**What it is.** {g['what_it_is']}",
        f"**When to get it checked.** {g['seek_care']}",
        "**What you can do now**\n" + "\n".join(f"- {x}" for x in g["self_care"]),
        "**See a doctor sooner if**\n" + "\n".join(f"- {x}" for x in g["red_flags"]),
        "---",
        f"_{care.DISCLAIMER} This result comes from a model that can be wrong._",
    ]
    return "\n\n".join(parts)


def _unsure_group_md(gp) -> str:
    e = care.EMERGENCY
    return (f"## Not sure what this is\n\nThe model could not tell confidently what kind of "
            f"thing this is (best guess: {GROUP_DISPLAY[gp.top_group]}, {gp.top_prob:.0%}). "
            f"That is not reassurance. If it worries you, is getting worse, or came on "
            f"suddenly, show it to a doctor or pharmacist.\n\n**{e['label']}:**\n"
            + "\n".join(f"- {x}" for x in e["signs"]) + f"\n\n---\n_{care.DISCLAIMER}_")


def _groups_facts(groups_dir: pathlib.Path) -> str:
    config = json.loads((groups_dir / "config.json").read_text())
    lines = [f"**Step 1 model (what kind of thing):** {config['arch']}, run `{config['run_id']}`, "
             f"{config['n_train']} training images (PAD-UFES-20 + SCIN phone photos)"]
    best = groups_dir / "best_val_report.json"
    if best.exists():
        lines.append(f"Best validation macro F1: {json.loads(best.read_text())['macro_f1']:.3f}")
    cal = groups_dir / "calibration.json"
    if cal.exists():
        c = json.loads(cal.read_text())
        t = c["test"]
        acc = "n/a" if t["accuracy_on_answered"] is None else f"{t['accuracy_on_answered']:.0%}"
        lines.append(f"Calibrated: threshold {c['threshold']:.2f}; on test it answered "
                     f"{t['coverage']:.0%} of images and was right on {acc} of those.")
    lines.append("Unfamiliar-image check: " + ("on" if (groups_dir / "ood.npz").exists() else "off"))
    return "\n\n".join(lines)


def build_app(run_dir: pathlib.Path, threshold: float | None,
              groups_dir: pathlib.Path | None = None) -> gr.Blocks:
    run = load_run(run_dir)
    calibrated = run.threshold is not None
    if threshold is None:
        threshold = run.threshold if calibrated else UNCALIBRATED_THRESHOLD
    grun = load_run(groups_dir) if groups_dir else None
    g_thr = (grun.threshold if grun and grun.threshold is not None else UNCALIBRATED_THRESHOLD)

    def lesion_outputs(image, thr):
        pred = predict(run, image, thr)
        types = {display_name(k): v for k, v in pred.probs.items()}
        triage = {k.capitalize(): v for k, v in pred.triage_probs.items()}
        if pred.unfamiliar:
            # Nothing about lesion types is shown: the guess would be meaningless.
            return (pred.heatmap,
                    gr.Label(value="Image not recognised", label="Triage"),
                    gr.Label(value=None, label="Lesion type (not shown -- image not recognised)"),
                    _guidance_md(pred, thr))
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

    def analyse(image, thr):
        if image is None:
            return None, None, None, None, "Upload a photo to begin."
        if grun is None:
            return lesion_outputs_with_kind(image, thr)
        gp = predict_group(grun, image, g_thr)
        kinds = {GROUP_DISPLAY[k]: v for k, v in gp.probs.items()}
        none_t = gr.Label(value=None, label="Triage (spots and growths only)")
        none_l = gr.Label(value=None, label="Lesion type (spots and growths only)")
        if gp.unfamiliar:
            u = care.UNFAMILIAR
            md = f"## {u['label']}\n\n{u['note']}\n\n---\n_{care.DISCLAIMER}_"
            return (gp.heatmap, gr.Label(value="Image not recognised", label="What kind of thing"),
                    none_t, none_l, md)
        if gp.abstained:
            return (gp.heatmap, gr.Label(value=kinds, label="What kind of thing (not confident)"),
                    none_t, none_l, _unsure_group_md(gp))
        if gp.top_group != "spot_growth":
            return (gp.heatmap, gr.Label(value=kinds, label="What kind of thing"),
                    none_t, none_l, _group_md(gp))
        heat, tri, typ, md = lesion_outputs(image, thr)
        return heat, gr.Label(value=kinds, label="What kind of thing"), tri, typ, md

    def lesion_outputs_with_kind(image, thr):
        heat, tri, typ, md = lesion_outputs(image, thr)
        return heat, gr.Label(value=None, label="What kind of thing (step 1 model not loaded)"), tri, typ, md

    with gr.Blocks(title="MEdAI") as app:
        gr.Markdown("# MEdAI\nA first check for things on your skin. "
                    "**Not a diagnosis.**\n\n" + (SCOPE_NOTE_GROUPS if grun else SCOPE_NOTE))
        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(type="pil", label="Photo of the skin")
                thr = gr.Slider(0.34, 0.99, value=threshold, step=0.01,
                                label=("Confidence needed to answer about spots and growths "
                                       + (f"(calibrated default {run.threshold:.2f})"
                                          if calibrated else "(uncalibrated placeholder)")))
                btn = gr.Button("Analyse", variant="primary")
                with gr.Accordion("About this model", open=False):
                    gr.Markdown((_groups_facts(groups_dir) + "\n\n---\n\n" if grun else "")
                                + "**Step 2 model (spots and growths):**\n\n" + _model_facts(run_dir))
            with gr.Column(scale=1):
                heat = gr.Image(label="Where the model looked (Grad-CAM; red = most influence)")
                kind = gr.Label(label="What kind of thing", num_top_classes=3)
                triage = gr.Label(label="Triage", num_top_classes=3)
                types = gr.Label(label="Lesion type", num_top_classes=4)
        guidance = gr.Markdown()

        outs = [heat, kind, triage, types, guidance]
        btn.click(analyse, [image, thr], outs, api_name="analyse")
        image.upload(analyse, [image, thr], outs, api_visibility="private")

        # JSON endpoint for the React interface (medai-mockup): /predict_json
        def api_predict(img):
            if img is None:
                return {"error": "no image"}
            return predict_json(run, grun, img, threshold, g_thr)

        with gr.Row(visible=False):
            api_in = gr.Image(type="pil")
            api_out = gr.JSON()
            api_btn = gr.Button()
        api_btn.click(api_predict, api_in, api_out, api_name="predict_json")
    return app


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="spots-and-growths run id under results/runs/, or a path")
    ap.add_argument("--groups-run", default=None,
                    help="optional Level 1 run (what kind of thing) -- see ADR 0007")
    ap.add_argument("--threshold", type=float, default=None,
                    help="override the run's calibrated threshold")
    ap.add_argument("--share", action="store_true", help="create a public gradio.live link")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="127.0.0.1",
                    help="0.0.0.0 to accept connections from other devices on the network")
    args = ap.parse_args()

    def resolve(r):
        d = pathlib.Path(r)
        return d if d.is_dir() else ROOT / "results" / "runs" / r

    run_dir = resolve(args.run)
    groups_dir = resolve(args.groups_run) if args.groups_run else None
    for d in [run_dir] + ([groups_dir] if groups_dir else []):
        if not (d / "best.pt").exists():
            print(f"no best.pt in {d} -- has training saved a checkpoint yet?")
            return 1

    build_app(run_dir, args.threshold, groups_dir).launch(
        share=args.share, server_port=args.port, server_name=args.host)
    return 0


if __name__ == "__main__":
    sys.exit(main())
