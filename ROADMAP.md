# Roadmap

Full reasoning in [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md). This is the checklist.

Roughly six months of calendar, about three months of real working time, with a dead
month for end-of-semester exams in the middle.

## Block A — now to exams (~4 weeks, part-time)
Work that tolerates being picked up and put down.

**1 · Data**
- [x] HAM10000 (~2.9 GB) — Harvard Dataverse `doi:10.7910/DVN/DBW86T`
- [x] PAD-UFES-20 (2,298 smartphone images) — ISIC collection 406
- [x] Documented label mapping between the two label sets (7 vs 6 classes)
- [x] Stratified splits, fixed seed, committed manifest
- [x] `docs/DATASET_CARD.md` — imbalance, domain split, skin-tone skew

**2 · Baseline**
- [ ] Experiment tracking wired up from run one
- [x] One lightweight model, two-stage fine-tune (freeze → unfreeze at low LR)
- [x] Balanced class weights
- [x] Class-wise metrics from run one

## Block B — exam month
Nothing scheduled. Long runs can sit in Colab unattended. This is the buffer.

## Block C — post-exams (~6 weeks, heavy)

**3 · The deployment gap, then the fix**  ← the centrepiece · results in README
- [x] Cold cross-domain evaluation: HAM10000-trained, PAD-tested, no adaptation
- [x] Phone-capture augmentation (blur, white balance, angle, JPEG artefacts)
- [x] Fine-tune on PAD-UFES-20; headline metrics reported on phone photos
- [x] Per-class before/after
- [ ] Train on both datasets together, so the phone-photo model stops over-flagging dermatoscope images (malignant precision 0.293 on HAM10000 test after PAD fine-tune)
- [ ] Separate prior shift from appearance shift in the cross-domain gap

**4 · Segmentation head**
- [ ] Lightweight U-Net or shared encoder; Dice and IoU
- [ ] Feeds Grad-CAM evaluation and lesion cropping

**5 · Calibration and abstention**
- [x] Temperature scaling + reliability diagram (bins saved in `calibration.json`; plot not yet drawn)
- [x] Abstention threshold chosen on validation against a target **malignant** recall (melanoma too rare in PAD-UFES-20 to set it on)
- [ ] Input-quality gate (blur / not-skin)

**6 · Explainability, measured**
- [x] Grad-CAM
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
- [x] Abstention becomes real behaviour, not a demo checkbox

## Block E — ship and defend (~4 weeks)

**10 · Deploy**
- [ ] Hugging Face Spaces (free CPU), prediction logging, basic drift check

**11 · Statistical rigour**
- [x] Bootstrap confidence intervals on headline metrics
- [ ] McNemar's test when claiming one model beats another

**12 · Write up**
- [ ] Model card, dataset card, README with real numbers
- [ ] Report written against measured results
