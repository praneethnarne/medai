# 7. Expand from skin-cancer screening to everyday skin concerns

Date: 2026-11-08

## Status
Accepted (product owner decision)

## Context
MEdAI was scoped as a skin-cancer screening aid: eight lesion types from HAM10000 and
PAD-UFES-20, collapsed to benign / indeterminate / malignant.

Two things changed the picture:

1. **Users do not know in advance what they are looking at.** People photograph a large
   insect bite, a sting, a cut that is healing oddly, a rash or a scar just as readily as a
   mole. A tool that only understands eight tumour-type lesions has nothing useful to say
   about most of what will be uploaded.
2. **The current model proves the point.** A healed scar received "actinic keratosis,
   book a doctor" from the calibrated model. An unfamiliar-image check
   (`backend/src/eval/ood.py`) now declines such images, but declining is not an answer to
   "should I worry about this?".

The question the product answers is "is this worth worrying about, and what should I do?"
— whatever is on the skin.

## Decision
Expand the scope to common everyday skin concerns, with skin-cancer screening as one part
of it. Structure the model in two levels:

- **Level 1 — what kind of thing is this:** spot or growth · rash or inflammation ·
  infection · bite or sting · wound, cut, burn or scar · normal skin · not skin / other.
- **Level 2 — which one:** the existing eight spot/growth types, plus the most common
  conditions in each Level 1 group for which enough labelled images exist.
- **Always — how urgent:** a triage tier and care guidance, with the existing rules: no
  drug names or dosages, referral leads for anything potentially serious, declining when
  not confident or not recognised.

Phone photos remain the primary domain (ADR 0004).

## Consequences
- New data sources are needed. Candidates: SCIN (Google/Stanford; crowd-sourced phone
  photos, mostly allergic, infectious and inflammatory conditions; SCIN Data Use License),
  Fitzpatrick17k (114 conditions; CC BY-NC-SA 3.0 labels, images fetched from source),
  and public wound sets (several on Roboflow Universe with CC BY 4.0 or public-domain
  labels for abrasions, bruises, burns and cuts). Each licence is checked before use and
  recorded in the dataset card.
- Labels from different sources disagree in wording and granularity; a documented mapping
  into the two levels is required, as ADR 0006 did for the ISIC sources.
- Accuracy per class will fall as classes are added, and some conditions (eczema,
  psoriasis, fungal infection) are hard to tell apart from a photo even for clinicians.
  Results are reported per level and per class with support counts and intervals, as now.
- The non-lesion catch rate, unmeasured today, becomes measurable once labelled
  non-lesion images exist.
- Care guidance must be written for every new condition and reviewed; urgency tiers are a
  clinical judgement and are flagged for review by the project guide.
- The skin-cancer results already measured (`README.md`, `results/eval/`) stand and remain
  the spot/growth branch of Level 2.
