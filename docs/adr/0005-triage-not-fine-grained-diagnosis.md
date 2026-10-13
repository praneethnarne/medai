# 5. Malignancy triage is the primary task; six-class diagnosis is secondary

Date: 2026-10-13

## Status
Accepted

## Context
An audit of PAD-UFES-20 (`docs/DATA_AUDIT.md`) found melanoma represented by 52 images out
of 2,298 — 2.3%. With patient-level splitting that leaves roughly five melanoma images in
the test set. Any per-class metric computed on five examples has a confidence interval wide
enough to be useless, and reporting it as a headline number would be misleading.

Grouped by malignancy the same dataset is well powered: 1,089 malignant, 730 indeterminate,
479 benign.

Separately, the product decision the system actually supports is not "which of six
diagnoses is this" — it is "does this need a doctor, and how soon".

## Decision
Make malignancy triage (malignant / indeterminate / benign) the primary task and the
headline metric. Keep six-class diagnosis as secondary, informational output.

Melanoma recall is still reported, but always with a bootstrap confidence interval and the
support count printed beside it.

## Consequences
The headline metric is well powered and matches what the interface actually tells the user,
which also makes the abstention threshold meaningful — it can be tuned against malignant
recall rather than against a near-empty melanoma class.

The project can no longer claim fine-grained diagnostic performance, and should not. The
six-class numbers will be reported with their support counts and treated as indicative.

HAM10000's 1,113 melanoma images remain valuable during pretraining, so the model still
sees melanoma in quantity even though the in-domain evaluation cannot measure it well.
