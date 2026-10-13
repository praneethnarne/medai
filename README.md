# MEdAI

Explainable AI for dermatological screening. Final-year project, MLR Institute of
Technology.

MEdAI classifies a dermatoscopic image into one of the seven HAM10000 lesion categories,
explains the prediction with a Grad-CAM heatmap, and turns the result into practical
care guidance.

## Status

Built from scratch over roughly two months. Nothing here is claimed until it is measured
— see `ROADMAP.md` for what is done and what is not.

| Component | Status |
|---|---|
| Dataset pipeline | not started |
| Baseline model | not started |
| Architecture comparison | not started |
| Grad-CAM | not started |
| Care guidance | drafted in the mockup |
| Backend API | not started |
| Frontend | mockup only — [praneethnarne/medai-mockup](https://github.com/praneethnarne/medai-mockup) |

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
