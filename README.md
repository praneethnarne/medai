# MEdAI

Explainable AI for dermatological screening. Final-year project, MLR Institute of
Technology.

> **New here, or picking this up on another machine? Read [START_HERE.md](START_HERE.md),
> then run `./setup.sh`.** It installs everything, downloads both datasets, builds the
> splits and runs the tests in one command.

MEdAI classifies a photo of a skin lesion into a triage level and one of eight lesion
types, explains the prediction with a Grad-CAM heatmap, declines to answer when it is not
confident, and turns the result into practical care guidance. It is built for phone
photos, not only dermatoscope images.

## Status

| Component | Status |
|---|---|
| Dataset pipeline | done — HAM10000 + PAD-UFES-20 from ISIC, verified, lesion/patient-level splits |
| Baseline model | done — EfficientNet-B0 on HAM10000 (`results/runs/ham_effb0_colab`) |
| Phone-photo fine-tune | done — PAD-UFES-20, with and without phone-capture augmentation |
| Calibration + abstention | done — temperature scaling, threshold chosen on validation |
| Grad-CAM | done in the demo; not yet measured against lesion masks |
| Care guidance | done — `backend/src/demo/care.py`, no drug names |
| Web demo | done — Gradio (`backend/src/demo/app.py`), launched from the Colab notebook |
| Architecture comparison | not started |
| Segmentation head | not started |
| Backend API | not started |

Training runs on a free Colab GPU: open `notebooks/train_on_colab.ipynb`.

## Results

All numbers are on held-out **test** sets, at the triage level (benign / indeterminate /
malignant), with 95% bootstrap confidence intervals. Each is read from the file named
beside it under `results/eval/`.

**Phone photos (PAD-UFES-20 test, 333 images)**

| Model | Accuracy | Macro F1 | Malignant recall | Source |
|---|---|---|---|---|
| Trained on dermatoscope images only | 0.450 [0.396, 0.505] | 0.380 [0.333, 0.426] | 0.509 | `ham_effb0_colab__pad_ufes_20_test__triage.json` |
| + fine-tuned on phone photos | 0.793 [0.748, 0.838] | 0.776 [0.725, 0.823] | 0.877 | `pad_ft_effb0__pad_ufes_20_test__triage.json` |
| + phone-capture augmentation (deployed) | **0.811** [0.769, 0.850] | **0.801** [0.755, 0.844] | **0.908** | `pad_ft_effb0_phoneaug__pad_ufes_20_test__triage.json` |

The augmented run was chosen on validation macro F1 (0.676 vs 0.614). On test its lead is
inside the confidence intervals, so the claim is "did not hurt", not "proven better".

**Dermatoscope images (HAM10000 test, 1,656 images)**

| Model | Accuracy | Macro F1 | Malignant recall | Malignant precision |
|---|---|---|---|---|
| Trained on dermatoscope images | 0.872 [0.855, 0.887] | 0.621 [0.562, 0.673] | 0.700 | 0.667 |
| Deployed phone-photo model | 0.598 [0.576, 0.620] | 0.392 [0.364, 0.421] | 0.729 | 0.293 |

The deployed model over-flags dermatoscope images as malignant — it learned PAD-UFES-20's
case mix (about half malignant). It is a phone-photo model; joint training is next.

**Calibration and abstention** (`pad_ft_effb0_phoneaug__pad_ufes_20_calibration.json`)

Temperature 0.80, fitted on validation; calibration error (ECE) 0.058 → 0.054 on test. The
demo declines below a triage confidence of **0.64**, chosen on validation so that at least
90% of malignant cases are flagged or referred. On test it answers 87.1% of images
[83.2%, 90.4%], is 84.8% accurate on those, and 94.5% of malignant cases [91.0%, 97.6%]
are flagged malignant or referred to a doctor.

**Known limits.** The phone-photo test set is small and holds few melanomas. The model
recognises eight lesion types only; scars, wounds and rashes are out of scope and there is
no dedicated "not a lesion" detector yet. HAM10000 is predominantly light-skinned.

## What makes this different

See `docs/DIFFERENTIATION.md`. In short: the system this project builds on maximises
accuracy through a four-model ensemble and reports a single number. MEdAI trades that
ensemble for a single lightweight model, measures what the trade actually costs, explains
each prediction, declines to answer when it is not confident, and tells the user what to
do next.

## Ground rules

1. **No number appears anywhere it was not measured.** Results are generated into
   `results/benchmark.json` and read from there — never typed into a paper, a report or
   a UI by hand.
2. **Every reported metric is reproducible** from a script in this repo.
3. **Class-wise metrics always accompany accuracy.** HAM10000 is heavily imbalanced;
   overall accuracy alone is misleading.
4. **Care guidance never prescribes treatment** — no drug names, no dosages. For
   malignant or pre-cancerous predictions, referral leads.

## Layout

```
backend/     inference API and training/evaluation scripts
frontend/    web interface
notebooks/   exploration — not the source of any reported result
docs/        design notes, dataset card, decision records
results/     generated metrics and figures
```

---

MEdAI is a screening aid, not a diagnosis. Only a qualified clinician can diagnose a
skin condition.
