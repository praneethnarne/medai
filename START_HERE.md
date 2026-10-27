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

## What is already built and working

| Area | State |
|---|---|
| Data acquisition | `backend/src/data/download.py` — both datasets from the ISIC API, parallel, resumable |
| Integrity | `verify.py` — checks JPEG start/end markers, not the manifest's size field (see below) |
| Labels | `labels.py` — ISIC's own taxonomy; triage (3 classes) and lesion type (8) |
| Splits | `split.py` — patient-level for PAD, lesion-level for HAM, leakage-tested |
| Dataset card | `docs/DATASET_CARD.md`, generated from the audits |
| Metrics | `train/metrics.py` — class-wise, bootstrap-ready, with the integrity gate |
| Training | `train/train.py` — two-stage fine-tune, **fully resumable** |
| Evaluation | `eval/evaluate.py` — bootstrap CIs, cross-domain support |
| Tests | 28 passing |

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

### 1. Finish the baseline  (~1-2 h on a GPU)

The last run reached full-stage epoch 9 of 15 before being stopped for unrelated
reasons; `results/runs/ham_baseline_effb0/history.json` has it. Checkpoints were
not committed (too large), so **start it fresh**:

```bash
./.venv/bin/python -u backend/src/train/train.py \
    --collection ham10000 --target lesion_type --arch efficientnet_b0 \
    --epochs-head 3 --epochs-full 15 --run-name ham_baseline_effb0
```

Or, to run it overnight on a laptop without the machine sleeping through it:

```bash
./train_overnight.sh
```

Stop any time with Ctrl-C; resume with the same command plus `--resume`. Expect
val macro F1 around 0.62-0.70.

### 2. The headline experiment  (~5 min)

```bash
# in-domain: how it does on the data it was trained on
./.venv/bin/python backend/src/eval/evaluate.py \
    --run ham_baseline_effb0 --collection ham10000 --split test

# cross-domain: the same model on real phone photos -- THE result
./.venv/bin/python backend/src/eval/evaluate.py \
    --run ham_baseline_effb0 --collection pad_ufes_20 --split test \
    --collapse-to-triage
```

The gap between those two numbers is what this project is about. Report it
whatever it says, then write it up in `docs/` and commit.

### 3. Then, in order

Fine-tune on PAD-UFES-20 so the model is trained for its actual domain, with
phone-capture augmentation evaluated as a hypothesis rather than assumed. Then
calibration and abstention, a segmentation head, measured Grad-CAM faithfulness,
the FastAPI backend, and the frontend. `ROADMAP.md` has the full sequence and
`docs/PROJECT_PLAN.md` the reasoning.

## Working agreement with the project owner

- **Never start a long training run unattended.** It was run on an 8 GB laptop
  during a live presentation and caused severe lag. Training starts only when the
  owner says they are away from the machine.
- Commit and push after each meaningful step; the repo is the only durable record.
- The public demo repo is `praneethnarne/medai-mockup` — a UI mockup with
  simulated inference. Do not confuse its numbers with real ones.
