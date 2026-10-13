# 2. Build from scratch rather than inherit the existing repositories

Date: 2026-10-13

## Status
Accepted

## Context
Four existing repositories were reviewed as a possible starting point. They form a
lineage rather than four alternatives. Findings:

- Grad-CAM — the project's central claimed contribution — does not exist in any of them.
- `src/accuracy.py` computes the ensemble accuracy, discards it, and prints a hardcoded
  `0.8189`. The same figure is served by the API as `MODEL_ACCURACY = 81.89`.
- The committed ensemble classification report states `accuracy 0.82` while its own
  weighted-average recall reads `0.72`. Those two numbers are identical in any real
  sklearn report. Recomputed from per-class recall and support, the true value is 0.718.
- The training scripts freeze the backbone and never unfreeze, so no fine-tuning stage
  ever ran. This is the likely reason accuracy plateaued around 0.69.
- The architecture is the reference paper's four-model soft-voting ensemble, which is
  what this project intends to differentiate itself from.

## Decision
Build from scratch. Keep the trained `.keras` weights from the old repositories as a
baseline to benchmark against, archived at `~/Developer/medai-legacy`.

## Consequences
More work up front, and no inherited fabricated results. Every number this project
reports will trace to a script in this repository. The old weights remain useful as a
comparison point for "what the previous approach achieved".
