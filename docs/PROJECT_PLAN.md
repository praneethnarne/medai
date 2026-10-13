# MEdAI — project plan

## What we are building, in one sentence

A skin-lesion screening aid that works on ordinary phone photos, explains which part of
the image drove its answer, and refuses to answer when it is not confident enough to be
useful.

## Why this framing

The standard version of this project — HAM10000 + a CNN + Grad-CAM + a web app — has been
published many times over. We are not trying to beat those papers; we are building
something that holds up when someone actually uses it, and saying honestly where it does
not.

The gap we build around is real and sits in plain sight: **almost every published system
trains and tests on dermatoscopic images, then claims to be a screening tool for the
public.** The public does not own a dermatoscope. They have a phone. Models trained on
dermoscopy degrade badly on consumer photos, and the application papers do not measure it.

We measure it, we mitigate what we can, and the system declines to answer when the input
is outside what it can handle. That is the project.

## Three pillars

1. **Works on phone photos, not just dermatoscopes.** Train on HAM10000 (dermatoscopic),
   then evaluate honestly on PAD-UFES-20 (2,298 real smartphone images). Report the drop.
   Then close some of it.
2. **Explains itself, and the explanation is checked.** Grad-CAM, plus a measurement of
   whether the heatmap actually lands on the lesion rather than on hair, ruler marks or
   bare skin.
3. **Knows when to shut up.** Calibrated confidence, an abstention threshold tuned around
   melanoma recall, and an input-quality gate. Below the bar it returns "get this looked
   at" instead of a label.

On top of those sits the care-guidance layer: triage tier, interim self-care, red flags.

## Constraints we are designing around

No budget, no paid compute, no paid hosting, no subscriptions. Everything below runs on
free tiers:

| Need | What we use |
|---|---|
| Training | Google Colab free tier (T4). One small model, ~1–2 h per run. |
| Datasets | HAM10000 and PAD-UFES-20 — both free, both already in the ISIC archive |
| Weights | Git LFS, or Hugging Face Hub (free) if LFS quota becomes a problem |
| Hosting | Hugging Face Spaces, free CPU tier |
| CI | GitHub Actions free minutes |

This will not be production grade and the README will say so. It will be *engineered*
like something intended for production, which is the part that matters here.

## Eight weeks

Each phase gets a proper plan when we reach it. Dates are targets, not promises.

**1 · Data.** Acquire HAM10000 and PAD-UFES-20. Stratified splits with a committed
manifest and a fixed seed. Write `docs/DATASET_CARD.md` recording class imbalance, the
dermatoscope/phone split, and the skin-tone skew we inherit and cannot fix.

**2 · Baseline.** One lightweight model (EfficientNet-B0 or MobileNetV3), trained
properly: balanced class weights *and* a two-stage fine-tune — freeze, then unfreeze the
top at a low learning rate. Class-wise metrics from the first run, never accuracy alone.

**3 · The deployment gap.** Evaluate the HAM10000-trained model on PAD-UFES-20 without
retraining. Record the drop — this number is the project's centrepiece and we publish it
whatever it says. Then mitigate: augmentation that simulates phone capture (blur, white
balance, angle, compression), and a small fine-tune on in-domain data. Report before and
after.

**4 · Knowing when to stop.** Temperature scaling for calibration, a reliability diagram,
an abstention threshold chosen against a target melanoma recall rather than picked by
eye, and a cheap input-quality gate for blurred or non-skin images.

**5 · Explainability, measured.** Grad-CAM, then energy-in-mask and IoU against ISIC
lesion masks on a held-out subset. Sanity test: the heatmap must change when the target
class changes.

**6 · Backend.** FastAPI — `/predict`, `/explain`, `/metrics`, `/health`. Pydantic
schemas, automatic OpenAPI docs, a Dockerfile, pytest covering the inference path, and
GitHub Actions running tests on every push.

**7 · Frontend.** Promote the mockup's screens onto real endpoints. The abstention path
stops being a checkbox and becomes real behaviour.

**8 · Ship and write up.** Deploy to Hugging Face Spaces. Model card, dataset card,
README with real numbers. Then the report, written against measured results.

## Ground rules

1. **No number appears anywhere it was not measured.** Metrics are generated into
   `results/benchmark.json` and read from there — never typed into a report by hand.
2. **Negative results get reported.** If phone-photo accuracy is poor, that is the
   finding, not something to hide. It is also the most interesting thing we will have
   to say.
3. **Class-wise metrics always accompany accuracy.** Melanoma recall is reported
   prominently, never buried in a macro average.
4. **Care guidance never prescribes treatment** — no drug names, no dosages. For
   malignant or pre-cancerous predictions, referral leads.

## What this gives you to talk about

Not "I got 94% on HAM10000." Instead: *I found that my model lost N points moving from
dermatoscope images to phone photos, built a gate that detects when an image is outside
what the model handles, calibrated the confidence so the threshold meant something, and
measured whether my explanations were pointing at the lesion or at a hair.*

That is a different conversation, and a better one.
