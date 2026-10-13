# 6. Use ISIC's normalised taxonomy rather than hand-mapping the two label sets

Date: 2026-10-13

## Status
Accepted

## Context
The plan assumed we would have to hand-write a mapping between HAM10000's seven classes
and PAD-UFES-20's six, deciding case by case what to do with classes that exist in one
dataset and not the other. That kind of mapping is error-prone and easy to get subtly
wrong — for instance HAM10000's "Pigmented benign keratosis" and PAD-UFES-20's "Seborrheic
keratosis" describe the same family but are worded differently.

Inspecting the manifests showed both collections are served by ISIC under one already
normalised diagnosis hierarchy:

- `diagnosis_1` — Benign / Indeterminate / Malignant
- `diagnosis_2` — eight categories, six of which occur in both datasets
- `diagnosis_3` — the finest level, and the level where the wording diverges

## Decision
Read ISIC's hierarchy instead of writing our own mapping.

- **Primary target:** `diagnosis_1` → triage (benign / indeterminate / malignant).
- **Secondary target:** `diagnosis_2` → lesion type. Six shared classes (nevus, benign
  keratosis, actinic keratosis, basal cell carcinoma, squamous cell carcinoma, melanoma)
  plus two that only HAM10000 has (vascular, dermatofibroma).
- **`diagnosis_3` is not used as a target.** It is precisely where the datasets disagree
  in wording, and reconciling it by hand is the error this decision avoids.

An unmapped value raises `UnknownLabel` rather than falling back to an "other" bucket, so
a change in the archive fails loudly instead of silently polluting training data.
`backend/tests/test_labels.py` asserts the mapping covers every value in both manifests.

## Consequences
The mapping problem largely disappears, and what remains is explicit: vascular lesions and
dermatofibromas can be learned during pretraining but can never be evaluated in the
phone-photo domain, because PAD-UFES-20 contains none.

The triage priors differ sharply between domains — HAM10000 is 80% benign, PAD-UFES-20 is
47% malignant — which reinforces the earlier finding that cross-domain degradation has a
prior-shift component as well as an appearance-shift component, and the two must be
separated before the headline number means anything.
