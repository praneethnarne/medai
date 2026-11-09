# Start here

You are picking up a final-year project mid-build. This file is the complete
context — read it, run `./setup.sh`, and continue from "What's next".

## What this project is

**MEdAI** — a skin-lesion screening aid that works on ordinary phone photos,
explains which part of the image drove its answer, and refuses to answer when it
is not confident enough to be useful.

Final-year project, MLR Institute of Technology. Submitted end of 4th year 2nd
semester. It is a portfolio piece, not a publication or a startup — there is no
budget, so everything runs on free tiers.

## The one idea that shapes everything

Almost every published system in this area trains and tests on **dermatoscopic**
images — taken through a magnifier pressed against the skin — then claims to be a
screening tool for the public. The public does not own a dermatoscope. They have
a phone.

So: **PAD-UFES-20 (2,298 real smartphone images) is the primary domain**, and
HAM10000 (11,720 dermatoscopic images) is the *pretraining corpus* we borrow
because PAD alone is too small to train on. Headline metrics are reported on
phone photos.

Read `docs/DIFFERENTIATION.md` for how this departs from the reference paper, and
`docs/adr/` for every significant decision with its reasoning.

## Ground rules — these are not negotiable

1. **No number appears anywhere it was not measured.** Metrics are generated into
   `results/` and read from there, never typed by hand. The codebase this project
   replaced shipped a hardcoded accuracy (`accur = 0.8189`) that overrode the
   computed one. `backend/src/train/metrics.py` enforces the identity
   accuracy == weighted-average recall, and a test feeds it that exact fabricated
   figure to prove it is rejected.
2. **Negative results get reported.** If phone-photo accuracy is poor, that is the
   finding — and the most interesting thing the project has to say.
3. **Class-wise metrics always accompany accuracy**, with support counts. The
   phone-photo test set contains **four melanoma images**; a recall computed on
   four examples cannot carry a conclusion and must never be presented as if it
   could.
4. **Care guidance never prescribes treatment** — no drug names, no dosages.

## What is already built and working  (state as of 9 Nov 2026)

| Area | State |
|---|---|
| Data | HAM10000 + PAD-UFES-20 from ISIC (`download.py`, `verify.py`, `split.py`); SCIN phone photos (`data/scin.py`, manifest + case-level splits committed) |
| Labels | `labels.py` — triage (3) and lesion type (8); `groups.py` — Level 1 groups (spot/growth, rash, infection, bite/sting, wound/scar) |
| Training | `train/train.py` — two-stage, resumable; `--phone-aug`; `--target group` for Level 1 |
| Evaluation | `eval/evaluate.py` (bootstrap CIs; group runs add a SCIN-only shortcut check), `eval/calibrate.py` (temperature + abstention threshold), `eval/ood.py` ("I don't recognise this image" check) |
| Demo | `demo/app.py` (Gradio) + `demo/api.py` (`/predict_json` for the React interface) + `demo/care.py` (guidance, no drug names) |
| Interface | `praneethnarne/medai-mockup` — React site; connects to the model server on its scan page ("Model server" box) |
| Colab | `notebooks/train_on_colab.ipynb`, steps 1–20 |
| Tests | 43 passing (`pytest backend/tests`; skip `test_splits_are_reproducible` — it rewrites the split files under newer scikit-learn) |

### Trained models (weights in the owner's Google Drive, `MyDrive/medai_runs/<run>/`)

| Run | What | Status |
|---|---|---|
| `ham_effb0_colab` | EfficientNet-B0 on HAM10000 | done |
| `pad_ft_effb0` | fine-tuned on PAD-UFES-20 phone photos | done |
| `pad_ft_effb0_phoneaug` | same + phone-capture augmentation, calibrated, ood check — **the deployed model** | done |
| `groups_effb0` | Level 1 (what kind of thing) on PAD-UFES-20 + SCIN | **not trained yet** — Colab GPU quota ran out; notebook steps 16–20, or on the Mac |

### Headline results (test sets, triage level, 95% CI) — details in README

- Phone photos, dermatoscope-only model: accuracy 0.450, macro F1 0.380, malignant recall 0.509.
- Phone photos, deployed model: accuracy 0.811 [0.769, 0.850], macro F1 0.801 [0.755, 0.844], malignant recall 0.908.
- Calibrated threshold 0.64: answers 87% of phone photos; 94.5% of malignant cases flagged or referred.
- Trade-off: deployed model over-flags dermatoscope images (malignant precision 0.293). Joint training is the fix.
- Unfamiliar-image check flags 3.6% of real phone-photo lesions, 14.6% of dermatoscope images; catch rate on scars/rashes/non-skin is not measured (no labelled set). It did reject the owner's scar photo and a non-skin photo in testing.

### Traps already found — do not rediscover these

- **ISIC's manifest lies about file size.** `files.full.size` is exactly 3,209
  bytes smaller than what S3 serves, for every image. Completeness is checked by
  JPEG markers instead. Do not "fix" this back to a size comparison.
- **Splits must be grouped.** PAD has `patient_id`, HAM only `lesion_id`. The
  audit reports `finest_split_unit` and the splitter reads it. Image-level
  splitting puts the same lesion on both sides and inflates every metric.
- **The priors are inverted between datasets.** HAM10000 is 80% benign;
  PAD-UFES-20 is 47% malignant. Cross-domain degradation therefore has two
  causes — appearance shift and prior shift — and they must be separated before
  the headline number means anything.
- **macOS python.org builds have no CA bundle.** `backend/src/bootstrap.py`
  points `SSL_CERT_FILE` at certifi. Import it first in any entry point.
- **Freezing the backbone is not enough.** The previous codebase froze it and
  never unfroze, plateauing near 0.69. Our own run confirms the cost: head-only
  reached macro F1 **0.346**; unfreezing reached **0.620** and was still climbing.

## What's next — pick up here

### Now: the presentation to the external assessor (Tue 10 Nov 2026)

Run everything on the owner's Mac (no Colab, no GPU needed — inference only):

1. Copy `MyDrive/medai_runs/pad_ft_effb0_phoneaug/` from Google Drive to
   `results/runs/pad_ft_effb0_phoneaug/` (needs `best.pt`, `config.json`,
   `calibration.json`, `ood.npz`).
2. Model server, in this repo:
   `./.venv/bin/pip install -q -r backend/requirements.txt gradio` then
   `./.venv/bin/python backend/src/demo/app.py --run pad_ft_effb0_phoneaug`
3. Interface, in `medai-mockup`: `npm install` then `npm run dev`; open
   http://localhost:5173 → New scan → Model server `http://127.0.0.1:7860` → Connect.
4. Test images (held-out test split, filename starts with the true label): the owner's
   Drive, `MyDrive/medai_demo_images/`.
5. Do one scan before the assessor arrives (first scan loads the model).

A 12-slide deck draft exists as a claude.ai artifact; it predates the fine-tune results
and needs updating with them (Monday night task).

### After the presentation (≈ 2 months, ADR 0007)

1. Train `groups_effb0` (notebook steps 16–20) and report step 18's SCIN-only table.
2. Joint training on HAM10000 + PAD-UFES-20 so dermatoscope images are not over-flagged.
3. Roadmap Block F: more conditions per group, normal-skin and not-skin classes,
   guidance reviewed by the guide.
4. Fix `test_splits_are_reproducible` so it does not overwrite committed splits.

## Working agreement with the project owner

- **Be concise with instructions.** Numbered steps, commands in their own blocks; explain
  only when asked or when it is genuinely needed.
- The owner decides product scope (see ADR 0007: everyday skin concerns, not only cancer).

- **Never start a long training run unattended.** It was run on an 8 GB laptop
  during a live presentation and caused severe lag. Training starts only when the
  owner says they are away from the machine.
- Commit and push after each meaningful step; the repo is the only durable record.
- The public demo repo is `praneethnarne/medai-mockup` — a UI mockup with
  simulated inference. Do not confuse its numbers with real ones.
