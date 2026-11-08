"""Level 1 of the two-level model (ADR 0007): what KIND of thing is on the skin.

    spot_growth   moles, keratoses, skin cancers -- everything PAD-UFES-20 holds
    rash          inflammation, allergy, eczema-type and other rashes
    infection     fungal, bacterial, viral skin infections and infestations
    bite_sting    insect bites and stings
    wound_scar    cuts, scrapes, burns, bruises, scars -- injury and its aftermath

The mapping below is written out condition by condition from SCIN's dermatologist
labels, so every inclusion can be read and argued with. Conditions not listed are
EXCLUDED, not guessed: some (purpura, vasculitis, lupus, lymphoma) do not belong to any
of these groups cleanly, and a wrong group label teaches the model the wrong thing.
`audit()` in scin.py reports what was excluded and how many images that cost.
"""

from __future__ import annotations

GROUPS = ("spot_growth", "rash", "infection", "bite_sting", "wound_scar")

DISPLAY = {
    "spot_growth": "Skin spot or growth",
    "rash": "Rash or skin inflammation",
    "infection": "Skin infection",
    "bite_sting": "Insect bite or sting",
    "wound_scar": "Wound, burn, bruise or scar",
}

# SCIN condition name (exactly as labelled) -> group
SCIN_TO_GROUP = {
    # --- rash / inflammation -------------------------------------------------
    "Eczema": "rash",
    "Allergic Contact Dermatitis": "rash",
    "Irritant Contact Dermatitis": "rash",
    "CD - Contact dermatitis": "rash",
    "Acute dermatitis, NOS": "rash",
    "Chronic dermatitis, NOS": "rash",
    "Acute and chronic dermatitis": "rash",
    "Acute dermatitis": "rash",
    "Acute vesicular dermatitis": "rash",
    "Urticaria": "rash",
    "Psoriasis": "rash",
    "Drug Rash": "rash",
    "Hypersensitivity": "rash",
    "Photodermatitis": "rash",
    "Pityriasis rosea": "rash",
    "Keratosis pilaris": "rash",
    "Lichen Simplex Chronicus": "rash",
    "Lichen planus/lichenoid eruption": "rash",
    "Prurigo nodularis": "rash",
    "Stasis Dermatitis": "rash",
    "Seborrheic Dermatitis": "rash",
    "Perioral Dermatitis": "rash",
    "Rosacea": "rash",
    "Acne": "rash",
    "Miliaria": "rash",
    "Intertrigo": "rash",
    "Granuloma annulare": "rash",
    "Pigmented purpuric eruption": "rash",
    "Xerosis": "rash",
    "Lichenified eczematous dermatitis": "rash",
    "Erythema multiforme": "rash",
    "Hidradenitis": "rash",
    # --- infection / infestation ---------------------------------------------
    "Folliculitis": "infection",
    "Tinea": "infection",
    "Tinea Versicolor": "infection",
    "Impetigo": "infection",
    "Herpes Zoster": "infection",
    "Herpes Simplex": "infection",
    "Viral Exanthem": "infection",
    "Scabies": "infection",
    "Verruca vulgaris": "infection",
    "Molluscum Contagiosum": "infection",
    "Abscess": "infection",
    "Cellulitis": "infection",
    "Ecthyma": "infection",
    "Infected eczema": "infection",
    "Skin infection": "infection",
    "Onychomycosis": "infection",
    "Cutaneous larva migrans": "infection",
    # --- bites and stings ------------------------------------------------------
    "Insect Bite": "bite_sting",
    # --- wounds, burns, bruises, scars -----------------------------------------
    "Scar Condition": "wound_scar",
    "Abrasion, scrape, or scab": "wound_scar",
    "Abrasion and/or friction burn of thigh without infection": "wound_scar",
    "wound/abrasion": "wound_scar",
    "Superficial wound of body region": "wound_scar",
    "Animal bite - wound": "wound_scar",
    "Burn of skin": "wound_scar",
    "Burn of forearm": "wound_scar",
    "Sunburn": "wound_scar",
    "Traumatic blister": "wound_scar",
    "Traumatic ulcer": "wound_scar",
    "Traumatic petechiae": "wound_scar",
    "O/E - ecchymoses present": "wound_scar",
    "Local infection of wound": "wound_scar",
    "Inflicted skin lesions": "wound_scar",
    # --- spots and growths that also appear in SCIN ------------------------------
    "Actinic Keratosis": "spot_growth",
    "SK/ISK": "spot_growth",
    "Keratosis": "spot_growth",
    "Cyst": "spot_growth",
    "Melanocytic Nevus": "spot_growth",
    "Basal Cell Carcinoma": "spot_growth",
    "SCC/SCCIS": "spot_growth",
    "Melanoma": "spot_growth",
    "Dermatofibroma": "spot_growth",
    "Hemangioma": "spot_growth",
}


def group_of_scin(condition: str) -> str | None:
    """None means excluded -- see the module docstring."""
    return SCIN_TO_GROUP.get(condition)
