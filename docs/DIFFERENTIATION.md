# How MEdAI differs from its reference work

Written early and deliberately, so the answer exists before anyone asks it.

## The reference paper

Talari, Nithin, Harshitha Yadav, Kushwant Kumar Reddy, Gnana Sai & Aswani,
"Deep Learning-Driven Skin Disease Detection", 2026 International Conference on Machine
Learning and Autonomous Systems (ICMLAS), Bangkok, 11–13 March 2026.
DOI [10.1109/ICMLAS67792.2026.11483712](https://doi.org/10.1109/ICMLAS67792.2026.11483712)

Its approach: ResNet50 + EfficientNetB0 + EfficientNetB3 + MobileNetV2, each producing
softmax probabilities, fused by decision-level **soft voting**, with preprocessing,
augmentation and imbalance-aware training. Motivated by dermatologist scarcity and by
misdiagnosis on darker skin tones.

## Where MEdAI diverges

| | Reference paper | MEdAI |
|---|---|---|
| Output | A label from the ensemble | Label, calibrated confidence, visual explanation, and care guidance |
| Models | All four fused | The same four **compared**; one deployed |
| Selection criterion | Accuracy | Accuracy against model size and inference latency |
| Explainability | Not addressed | Grad-CAM as a core component |
| Uncertainty | Not addressed | Below a confidence threshold the system declines to give a label |
| After the prediction | Nothing | Triage tier, interim self-care, and red flags |
| Delivery | "Can be incorporated into clinical workflow" | Deployed web application |

## The argument in one paragraph

An ensemble of four models maximises accuracy and tells the user nothing about why it
decided what it decided, whether it can run inside a real request cycle, or what they
should do with the answer. MEdAI trades the ensemble for a single lightweight model,
measures exactly what that trade costs in accuracy and saves in size and latency,
explains each prediction with Grad-CAM, refuses to answer when it is not confident
enough to be useful, and converts the result into action. The contribution is not a
higher number — it is a system that is usable and honest.

## What we inherit, including the problems

- **HAM10000's class imbalance.** `nv` dominates. Accuracy alone is misleading, so
  class-wise metrics are reported alongside it everywhere.
- **HAM10000's skin-tone skew.** The reference paper raises darker-skin misdiagnosis as
  motivation but trains on a dataset that is overwhelmingly light-skinned. MEdAI inherits
  that limitation. We state it rather than claim to have solved it.
- **Minority-class performance.** Melanoma is the clinically critical class and the hardest
  one. Any honest evaluation reports its recall prominently rather than burying it in a
  macro average.
