# MEdAI — project plan

## What we are building, in one sentence

A skin-lesion screening system that works on ordinary phone photos, explains which part
of the image drove its answer, says how certain it is, and refuses to answer when it is
not confident enough to be useful.

## Why this framing

The standard version of this project — HAM10000 + a CNN + Grad-CAM + a web app — has been
published many times over. We are not trying to beat those papers. We are building
something that holds up when someone actually uses it, and saying honestly where it does
not.

The gap we build around is real and sits in plain sight: **almost every published system
trains and tests on dermatoscopic images, then claims to be a screening tool for the
public.** The public does not own a dermatoscope. They have a phone. Models trained on
dermoscopy degrade badly on consumer photos, and the application papers do not measure it.

We measure it, we mitigate what we can, and the system declines to answer when the input
is outside what it can handle.

## Scope: four pillars

**1 · Phone photos are the primary domain, not a stress test.**
Real users photograph a mole with a phone. Nobody owns a dermatoscope, and nobody takes a
DSLR shot, transfers it to a laptop and uploads it. So the domain that counts is
PAD-UFES-20 (2,298 real smartphone images), and HAM10000's 10,015 dermatoscopic images are
the *pretraining corpus* we borrow because PAD alone is too small to train on.

Pipeline: pretrain on HAM10000, fine-tune on PAD-UFES-20, and **report headline metrics on
phone photos**, with dermatoscopic numbers alongside for comparison. The cold cross-domain
number — HAM10000-trained, PAD-tested, no adaptation — is still measured, because it
quantifies how misleading dermoscopy-only results are when applied to the use case those
papers claim.

Two tasks fall out of this and must not be hand-waved. The label sets differ:
PAD-UFES-20 has six classes (BCC, SCC, ACK, SEK, MEL, NEV) against HAM10000's seven, so a
documented mapping is needed with the non-overlapping classes handled explicitly rather
than quietly dropped. And phone-capture augmentation — blur, white balance, angle, JPEG
artefacts — is evaluated on whether it actually earns its keep.

**2 · Two heads, not one.** A segmentation head alongside the classifier, so the system
outputs *where* the lesion is as well as *what* it is. This is not decoration: the mask
gives us ground truth for checking whether Grad-CAM points at the lesion, it lets us crop
to the lesion before classifying, and it is a visible second capability.

**3 · Explanations that are checked, not asserted.** Grad-CAM, plus energy-in-mask and
IoU against lesion masks. Every paper prints heatmaps; almost none measure whether they
land on the lesion rather than on a hair or a ruler mark. Ours are measured and the number
is reported whatever it says.

**4 · Knows when to shut up.** Temperature scaling for calibration, a reliability diagram,
an abstention threshold chosen against a target melanoma recall rather than picked by eye,
and an input-quality gate for blurred or non-skin images.

Sitting on top: image + clinical metadata fusion (PAD-UFES-20 ships 22 patient features,
HAM10000 has age, sex and body site), and the care-guidance layer with triage tiers.

## Constraints

No budget. No paid compute, hosting or subscriptions. Everything runs on free tiers:

| Need | What we use |
|---|---|
| Training | Google Colab free tier (T4) |
| Datasets | HAM10000, PAD-UFES-20 — both free |
| Experiment tracking | Weights & Biases free tier |
| Weights | Git LFS, or Hugging Face Hub if LFS quota bites |
| Hosting | Hugging Face Spaces, free CPU |
| CI | GitHub Actions free minutes |

This will not be production grade and the README will say so. It will be *engineered*
like something intended for production, which is the part that matters.

## Schedule

Roughly six months of calendar, about three months of real working time, with a dead
month for end-of-semester exams. The plan front-loads work that survives interruption and
puts the demanding work after exams.

### Block A — now to exams (~4 weeks, part-time)
Work that tolerates being picked up and put down.

1. **Data.** HAM10000 and PAD-UFES-20 acquired, stratified splits with a committed
   manifest and fixed seed, `docs/DATASET_CARD.md` covering class imbalance, the
   dermatoscope/phone domain split, and the skin-tone skew we inherit and cannot fix.
2. **Baseline.** One lightweight model, trained properly: balanced class weights *and* a
   two-stage fine-tune (freeze, then unfreeze the top at a low learning rate). Class-wise
   metrics from the first run, never accuracy alone. Experiment tracking wired up from
   the start so later runs are comparable.

### Block B — exam month (minimal)
No new work scheduled. Long training runs can sit in Colab unattended. Treat anything
achieved here as a bonus, and treat this block as the schedule's buffer.

### Block C — post-exams (~6 weeks, the heavy push)
3. **The deployment gap, then the fix.** Evaluate the HAM10000-trained model on
   PAD-UFES-20 cold — that number gets reported whatever it says. Then fine-tune on
   PAD-UFES-20 so the model is trained for the domain it will actually be used in, and
   report headline metrics on phone photos, per class, before and after.
4. **Segmentation head.** Lightweight U-Net or a shared-encoder multi-task setup. Dice and
   IoU reported. Feeds pillars 2 and 3.
5. **Calibration and abstention.** Temperature scaling, reliability diagram, threshold
   selection against a melanoma-recall target, input-quality gate.
6. **Explainability, measured.** Grad-CAM, energy-in-mask and IoU against the masks from
   phase 4. Sanity test: the heatmap must change when the target class is forced to change.

### Block D — build it properly (~4 weeks)
7. **Metadata fusion.** Image features plus patient metadata. Report whether it actually
   helps — a negative result here is still a result.
8. **Backend.** FastAPI: `/predict`, `/explain`, `/metrics`, `/health`. Pydantic schemas,
   automatic OpenAPI docs, Dockerfile, pytest over the inference path, GitHub Actions
   running tests and lint on every push.
9. **Frontend.** The mockup's screens promoted onto real endpoints. Abstention stops being
   a demo checkbox and becomes real behaviour.

### Block E — ship and defend (~4 weeks)
10. **Deploy** to Hugging Face Spaces, with prediction logging and a basic drift check —
    which is the deployment theme closing the loop on itself.
11. **Statistical rigour.** Bootstrap confidence intervals on the headline metrics, and
    McNemar's test when claiming one model beats another. Cheap to do, and it is the
    difference between "94%" and "94% (95% CI 92.1–95.6)".
12. **Write up.** Model card, dataset card, README with real numbers, then the report
    written against measured results.

Block E leaves slack before submission. It will get used.

## Ground rules

1. **No number appears anywhere it was not measured.** Metrics are generated into
   `results/benchmark.json` and read from there — never typed into a report by hand.
2. **Negative results get reported.** If phone-photo accuracy is poor, or metadata fusion
   does not help, that is the finding. It is also the most interesting thing we will have
   to say.
3. **Class-wise metrics always accompany accuracy.** Melanoma recall is reported
   prominently, never buried in a macro average.
4. **Care guidance never prescribes treatment** — no drug names, no dosages. For malignant
   or pre-cancerous predictions, referral leads.
5. **Every experiment is tracked and reproducible** from a script in this repository.

## What stops this being a toy

By submission the repository contains: two datasets with a written dataset card; a
multi-task model doing classification and segmentation; a measured cross-domain
generalisation study; a calibrated, abstaining classifier with a justified threshold;
quantitatively evaluated explanations; metadata fusion with an honest verdict; a typed,
containerised, tested API with CI; a deployed live demo; a model card; statistical
confidence intervals on every headline claim; and a set of dated architecture decision
records explaining why each choice was made.

The talking point is not "I got 94% on HAM10000" — every tutorial says that. It is: *I
found my model lost N points moving from dermatoscope images to phone photos, built a gate
that detects inputs it cannot handle, calibrated the confidence so the threshold meant
something, and measured whether my explanations pointed at the lesion or at a hair.*
