# Roadmap

Eight phases, each planned in detail when we reach it rather than guessed at now.

## 1 · Dataset
- [ ] Acquire HAM10000 (~2.9 GB) — Harvard Dataverse `doi:10.7910/DVN/DBW86T`
- [ ] Stratified train/val/test split, fixed seed, committed split manifest
- [ ] `docs/DATASET_CARD.md` — class distribution, provenance, and the skin-tone skew

## 2 · Baseline
- [ ] One lightweight model trained properly
- [ ] Balanced class weights **and** a two-stage fine-tune (freeze → unfreeze at low LR)
- [ ] Class-wise metrics from the first run, not just accuracy

## 3 · Architecture comparison
- [ ] Accuracy × model size × inference latency across candidate architectures
- [ ] Generated into `results/benchmark.json` as the single source of truth
- [ ] Integrity check: reported accuracy must equal weighted-average recall

## 4 · Grad-CAM
- [ ] Heatmap from last conv layer activations and class gradients
- [ ] Sanity test: heatmap changes when the target class is forced to change
- [ ] Overlay rendering

## 5 · Care guidance
- [ ] Condition knowledge base, reviewed against the safety rules in the README
- [ ] Ideally sanity-checked by the project guide or a clinician

## 6 · Backend
- [ ] FastAPI: `/predict`, `/explain`, `/metrics`
- [ ] Confidence threshold below which the API declines to give a label

## 7 · Frontend
- [ ] Promote the mockup's screens onto real endpoints

## 8 · Evaluation and paper
- [ ] Rewrite the IEEE paper against measured results
- [ ] Rebuild the results tables from `results/benchmark.json`
- [ ] Limitations section: class imbalance, dataset skin-tone skew, minority-class performance
