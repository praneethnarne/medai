"""The label space shared by HAM10000 and PAD-UFES-20.

Both collections are served by the ISIC archive, which has already normalised
them onto one diagnosis taxonomy. That saves us hand-rolling a mapping between
HAM10000's seven classes and PAD-UFES-20's six -- we read ISIC's own hierarchy
instead, at the level where the two datasets actually line up.

  diagnosis_1  ->  TRIAGE      benign / indeterminate / malignant  (primary task)
  diagnosis_2  ->  LESION_TYPE six shared classes + two HAM-only   (secondary)

`diagnosis_3` is deliberately not used as a target. It is where the two datasets
diverge in wording for the same thing -- HAM10000 says "Pigmented benign
keratosis" where PAD-UFES-20 says "Seborrheic keratosis" -- and reconciling that
by hand is exactly the error-prone step this module avoids.
"""

from __future__ import annotations

# --- Primary task: triage ---------------------------------------------------
# What the interface actually tells the user: does this need a doctor?

TRIAGE = ("benign", "indeterminate", "malignant")

_TRIAGE_FROM_ISIC = {
    "Benign": "benign",
    "Indeterminate": "indeterminate",
    "Malignant": "malignant",
}

# --- Secondary task: lesion type -------------------------------------------
# Six classes present in both datasets, then two that only HAM10000 has.

SHARED_TYPES = (
    "nevus",
    "benign_keratosis",
    "actinic_keratosis",
    "basal_cell_carcinoma",
    "squamous_cell_carcinoma",
    "melanoma",
)
HAM_ONLY_TYPES = ("vascular", "dermatofibroma")
LESION_TYPES = SHARED_TYPES + HAM_ONLY_TYPES

_TYPE_FROM_ISIC = {
    "Benign melanocytic proliferations": "nevus",
    "Benign epidermal proliferations": "benign_keratosis",
    "Indeterminate epidermal proliferations": "actinic_keratosis",
    "Malignant adnexal epithelial proliferations - Follicular": "basal_cell_carcinoma",
    "Malignant epidermal proliferations": "squamous_cell_carcinoma",
    "Malignant melanocytic proliferations (Melanoma)": "melanoma",
    "Benign soft tissue proliferations - Vascular": "vascular",
    "Benign soft tissue proliferations - Fibro-histiocytic": "dermatofibroma",
}

DISPLAY_NAMES = {
    "nevus": "Mole (nevus)",
    "benign_keratosis": "Benign keratosis",
    "actinic_keratosis": "Actinic keratosis",
    "basal_cell_carcinoma": "Basal cell carcinoma",
    "squamous_cell_carcinoma": "Squamous cell carcinoma",
    "melanoma": "Melanoma",
    "vascular": "Vascular lesion",
    "dermatofibroma": "Dermatofibroma",
}


class UnknownLabel(ValueError):
    """Raised when a record carries a diagnosis the mapping does not cover.

    Deliberately fatal. A silent fallback to an "other" bucket would let the
    archive add a category and have it quietly disappear into training data.
    """


def triage_of(record: dict) -> str:
    raw = record["metadata"]["clinical"].get("diagnosis_1")
    try:
        return _TRIAGE_FROM_ISIC[raw]
    except KeyError:
        raise UnknownLabel(f"unmapped diagnosis_1: {raw!r}") from None


def lesion_type_of(record: dict) -> str:
    raw = record["metadata"]["clinical"].get("diagnosis_2")
    try:
        return _TYPE_FROM_ISIC[raw]
    except KeyError:
        raise UnknownLabel(f"unmapped diagnosis_2: {raw!r}") from None


def is_shared_type(record: dict) -> bool:
    """True when this record's lesion type exists in both datasets.

    Vascular lesions and dermatofibromas appear only in HAM10000, so they can be
    learned during pretraining but never evaluated in the phone-photo domain.
    """
    return lesion_type_of(record) in SHARED_TYPES


# --- Collapsing lesion type to triage ---------------------------------------
# Each lesion type sits at exactly one triage level, verified against every
# record in both collections (see tests). This is what lets a model trained on
# HAM10000's eight lesion types produce triage predictions on PAD-UFES-20,
# whose label set is different -- the cross-domain experiment depends on it.

TRIAGE_OF_TYPE = {
    "nevus": "benign",
    "benign_keratosis": "benign",
    "dermatofibroma": "benign",
    "vascular": "benign",
    "actinic_keratosis": "indeterminate",
    "basal_cell_carcinoma": "malignant",
    "squamous_cell_carcinoma": "malignant",
    "melanoma": "malignant",
}


def triage_of_type(lesion_type: str) -> str:
    try:
        return TRIAGE_OF_TYPE[lesion_type]
    except KeyError:
        raise UnknownLabel(f"no triage level for lesion type {lesion_type!r}") from None
