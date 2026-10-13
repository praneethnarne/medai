# Roadmap

Full reasoning in [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md). This is the checklist.

Roughly six months of calendar, about three months of real working time, with a dead
month for end-of-semester exams in the middle.

## Block A — now to exams (~4 weeks, part-time)
Work that tolerates being picked up and put down.

**1 · Data**
- [ ] HAM10000 (~2.9 GB) — Harvard Dataverse `doi:10.7910/DVN/DBW86T`
- [ ] PAD-UFES-20 (2,298 smartphone images) — ISIC collection 406
- [ ] Documented label mapping between the two label sets (7 vs 6 classes)
- [ ] Stratified splits, fixed seed, committed manifest
- [ ] `docs/DATASET_CARD.md` — imbalance, domain split, skin-tone skew

**2 · Baseline**
- [ ] Experiment tracking wired up from run one
- [ ] One lightweight model, two-stage fine-tune (freeze → unfreeze at low LR)
- [ ] Balanced class weights
- [ ] Class-wise metrics from run one

## Block B — exam month
Nothing scheduled. Long runs can sit in Colab unattended. This is the buffer.

## Block C — post-exams (~6 weeks, heavy)

**3 · The deployment gap, then the fix**  ← the centrepiece
- [ ] Cold cross-domain evaluation: HAM10000-trained, PAD-tested, no adaptation
- [ ] Phone-capture augmentation (blur, white balance, angle, JPEG artefacts)
- [ ] Fine-tune on PAD-UFES-20; headline metrics reported on phone photos
- [ ] Per-class before/after

**4 · Segmentation head**
- [ ] Lightweight U-Net or shared encoder; Dice and IoU
- [ ] Feeds Grad-CAM evaluation and lesion cropping

**5 · Calibration and abstention**
- [ ] Temperature scaling + reliability diagram
- [ ] Abstention threshold chosen against a target melanoma recall
- [ ] Input-quality gate (blur / not-skin)

**6 · Explainability, measured**
- [ ] Grad-CAM
- [ ] Energy-in-mask and IoU against lesion masks
- [ ] Sanity test: heatmap changes when target class is forced

## Block D — build it properly (~4 weeks)

**7 · Metadata fusion**
- [ ] Image features + patient metadata; honest verdict on whether it helps

**8 · Backend**
- [ ] FastAPI: `/predict`, `/explain`, `/metrics`, `/health`
- [ ] Pydantic schemas, Dockerfile, pytest on the inference path
- [ ] GitHub Actions on every push

**9 · Frontend**
- [ ] Mockup screens promoted onto real endpoints
- [ ] Abstention becomes real behaviour, not a demo checkbox

## Block E — ship and defend (~4 weeks)

**10 · Deploy**
- [ ] Hugging Face Spaces (free CPU), prediction logging, basic drift check

**11 · Statistical rigour**
- [ ] Bootstrap confidence intervals on headline metrics
- [ ] McNemar's test when claiming one model beats another

**12 · Write up**
- [ ] Model card, dataset card, README with real numbers
- [ ] Report written against measured results
