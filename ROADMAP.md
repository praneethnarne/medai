# Roadmap

Full reasoning in [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md). This is the checklist.

## 1 · Data
- [ ] HAM10000 (~2.9 GB) — Harvard Dataverse `doi:10.7910/DVN/DBW86T`
- [ ] PAD-UFES-20 (smartphone images) — ISIC collection 406
- [ ] Stratified splits, fixed seed, committed manifest
- [ ] `docs/DATASET_CARD.md` — imbalance, domain split, skin-tone skew

## 2 · Baseline
- [ ] One lightweight model, two-stage fine-tune (freeze → unfreeze at low LR)
- [ ] Balanced class weights
- [ ] Class-wise metrics from run one

## 3 · The deployment gap  ← the centrepiece
- [ ] Evaluate HAM10000-trained model on PAD-UFES-20 cold. Record the drop.
- [ ] Phone-capture augmentation (blur, white balance, angle, JPEG compression)
- [ ] Small in-domain fine-tune; report before/after

## 4 · Knowing when to stop
- [ ] Temperature scaling + reliability diagram
- [ ] Abstention threshold chosen against a target melanoma recall
- [ ] Input-quality gate (blur / not-skin detection)

## 5 · Explainability, measured
- [ ] Grad-CAM
- [ ] Energy-in-mask and IoU vs ISIC lesion masks
- [ ] Sanity test: heatmap changes when target class changes

## 6 · Backend
- [ ] FastAPI: `/predict`, `/explain`, `/metrics`, `/health`
- [ ] Dockerfile, pytest on the inference path, GitHub Actions CI

## 7 · Frontend
- [ ] Promote mockup screens onto real endpoints
- [ ] Abstention becomes real behaviour, not a demo checkbox

## 8 · Ship and write up
- [ ] Deploy to Hugging Face Spaces (free CPU)
- [ ] Model card, dataset card, README with real numbers
- [ ] Report written against measured results
