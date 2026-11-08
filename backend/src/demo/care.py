"""Care guidance for each lesion type the model can predict.

Ported from the mockup (praneethnarne/medai-mockup, src/data/conditions.js) and
re-keyed to this repo's eight lesion types. Squamous cell carcinoma is new here:
the mockup used HAM10000's combined "akiec" class, which ISIC's taxonomy splits
into actinic keratosis and squamous cell carcinoma.

Content rules (deliberate, do not relax):
  - No drug names, no dosages, no prescriptive treatment.
  - For urgent and prompt tiers the referral leads; self-care is framed only as
    what to do while waiting for an appointment, never as an alternative.
  - Guidance follows a prediction that can be wrong, so it is always presented
    as conditional, never as a verdict.

The urgency tier per lesion type is a clinical judgement, not a model output.
It should be reviewed by the project guide before the system is used by anyone.
"""

from __future__ import annotations

DISCLAIMER = (
    "MEdAI is a first check, not a diagnosis. Only a qualified doctor can "
    "diagnose a skin condition."
)

URGENCY = {
    "urgent": {
        "label": "See a skin doctor urgently",
        "note": "This could be serious. Do not wait to see if it changes.",
    },
    "prompt": {
        "label": "Book a skin doctor appointment",
        "note": "A doctor needs to look at it, but it is not an emergency.",
    },
    "routine": {
        "label": "Monitor and mention at your next visit",
        "note": "Usually harmless. Keep an eye on it.",
    },
}

# Shown when the image does not look like any lesion the model was trained on.
UNFAMILIAR = {
    "label": "I don't recognise this image",
    "note": (
        "This photo does not look like the skin lesions MEdAI was trained on, so it "
        "gives no answer. It may be a scar, wound, rash, bite, another skin condition, "
        "or a photo that is too blurred, dark or far away. If you are worried about "
        "it, show it to a doctor."
    ),
}

# Shown when the model declines to answer.
UNSURE = {
    "label": "No confident answer -- please see a doctor",
    "note": (
        "The model is not confident enough about this image to say anything "
        "useful. That is not reassurance. If the spot worries you, or it has "
        "changed, have a doctor look at it."
    ),
}

UNIVERSAL_RED_FLAGS = [
    "It grows or changes shape over weeks rather than months",
    "It bleeds, weeps, crusts or will not heal",
    "Its colour becomes uneven, or new colours appear",
    "It becomes itchy, tender or painful",
]

CONDITIONS = {
    "melanoma": {
        "plain": "A dangerous kind of mole. This is the one that must not be ignored.",
        "urgency": "urgent",
        "what_it_is": (
            "Melanoma is a cancer that starts in the cells which give skin its "
            "colour. It is the most serious skin cancer, but when it is caught "
            "early it is usually very treatable. That is why speed matters here."
        ),
        "seek_care": (
            "Arrange a dermatologist or skin-cancer clinic appointment as soon as "
            "you can -- within days, not months."
        ),
        "self_care": [
            "Do not scratch, pick, shave over or try to remove the spot",
            "Photograph it next to a ruler or coin so the clinician can see its size",
            "Keep it covered from the sun, and avoid sunbeds completely",
            "Write down when you first noticed it and what has changed since",
        ],
        "red_flags": UNIVERSAL_RED_FLAGS,
    },
    "squamous_cell_carcinoma": {
        "plain": (
            "A common skin cancer that grows on sun-exposed skin. It needs to be "
            "seen by a doctor soon."
        ),
        "urgency": "urgent",
        "what_it_is": (
            "Squamous cell carcinoma is a skin cancer that usually appears as a "
            "scaly, crusted or raised patch on skin that gets a lot of sun. Most "
            "are cured when treated early, but unlike basal cell carcinoma it can "
            "occasionally spread, so it should not be left."
        ),
        "seek_care": (
            "Arrange a dermatologist or skin-cancer clinic appointment as soon as "
            "you can -- within days to two weeks."
        ),
        "self_care": [
            "Do not pick at or try to remove any crust or scab",
            "Keep it clean and covered from the sun",
            "Photograph it next to a ruler or coin so the clinician can see its size",
            "Note whether it has grown, bled or become tender",
        ],
        "red_flags": UNIVERSAL_RED_FLAGS,
    },
    "basal_cell_carcinoma": {
        "plain": (
            "The most common skin cancer. It grows slowly and almost never spreads, "
            "but a doctor does need to remove it."
        ),
        "urgency": "prompt",
        "what_it_is": (
            "Basal cell carcinoma is the most common skin cancer. It grows slowly "
            "and almost never spreads to the rest of the body, but it will not go "
            "away on its own and needs a doctor to treat it."
        ),
        "seek_care": (
            "Book a skin doctor appointment in the next few weeks. Earlier "
            "treatment usually means a smaller procedure."
        ),
        "self_care": [
            "Leave the area alone -- do not pick at any crust or scab",
            "Keep it clean and dry, and protect it from sun exposure",
            "Photograph it monthly so you can show any change",
            "Note whether it has bled after minor knocks or washing",
        ],
        "red_flags": UNIVERSAL_RED_FLAGS,
    },
    "actinic_keratosis": {
        "plain": (
            "A rough, sun-damaged patch. Not cancer, but a small number turn into "
            "cancer if left alone."
        ),
        "urgency": "prompt",
        "what_it_is": (
            "These are dry, rough patches caused by years of sun on the skin. They "
            "are not cancer, but a small number of them turn into cancer over "
            "time, so doctors treat them early while it is simple."
        ),
        "seek_care": (
            "Book a dermatologist or GP appointment in the next few weeks. These "
            "are very treatable at this stage, which is the reason to act now."
        ),
        "self_care": [
            "Protect the area from sun -- sunscreen, hats, long sleeves",
            "Avoid sunbeds completely",
            "Do not scrub, exfoliate or pick at the rough patch",
            "Check whether you have similar patches elsewhere and mention them too",
        ],
        "red_flags": UNIVERSAL_RED_FLAGS
        + ["The patch becomes thickened, lumpy or firm to the touch"],
    },
    "benign_keratosis": {
        "plain": "A harmless age-and-sun spot. Very common, and it does not turn into cancer.",
        "urgency": "routine",
        "what_it_is": (
            "These are harmless spots that build up with age and sun exposure. "
            "People often mistake them for something serious because they can "
            "look dark and crusty."
        ),
        "seek_care": (
            "No urgent action needed. Mention it at your next routine appointment, "
            "particularly if it bothers you or catches on clothing."
        ),
        "self_care": [
            "Leave it alone -- these do not need removing",
            "Use sunscreen on the area to stop further pigment change",
            "Do not try to scrape or freeze it off yourself",
            "Photograph it every few months to confirm it stays stable",
        ],
        "red_flags": UNIVERSAL_RED_FLAGS,
    },
    "nevus": {
        "plain": "An ordinary mole. Most adults have many of them.",
        "urgency": "routine",
        "what_it_is": (
            "This is an ordinary mole: a small cluster of the cells that give skin "
            "its colour. Nearly everyone has them, and the vast majority never "
            "cause any trouble."
        ),
        "seek_care": (
            "No action needed now. Have it looked at if it starts to change, and "
            "mention it at a routine skin check if you have many moles or a family "
            "history of melanoma."
        ),
        "self_care": [
            "Learn this mole's normal appearance so you notice change early",
            "Photograph it every few months from the same distance",
            "Use sunscreen and avoid burning",
            "Check your skin for any mole that looks different from your others",
        ],
        "red_flags": UNIVERSAL_RED_FLAGS,
    },
    "dermatofibroma": {
        "plain": "A harmless small lump in the skin, often left after an insect bite or scratch.",
        "urgency": "routine",
        "what_it_is": (
            "A dermatofibroma is a small, firm lump in the skin, usually left "
            "behind after a minor injury. It is harmless and tends to stay the "
            "same for years."
        ),
        "seek_care": (
            "No urgent action needed. Mention it at a routine appointment if it is "
            "painful, catches on clothing, or seems to be changing."
        ),
        "self_care": [
            "Leave it alone -- it needs no treatment",
            "Avoid shaving directly over it",
            "Note its size now so you can tell if it genuinely grows",
            "Protect it from repeated friction or knocks",
        ],
        "red_flags": UNIVERSAL_RED_FLAGS,
    },
    "vascular": {
        "plain": "A harmless red spot made of tiny blood vessels bunched together.",
        "urgency": "routine",
        "what_it_is": (
            "This is a cluster of tiny blood vessels close to the surface of the "
            "skin, which is what makes it look red. It is very common and it is "
            "not a cancer."
        ),
        "seek_care": (
            "No urgent action needed. See a clinician if it bleeds repeatedly, "
            "grows quickly, or keeps getting knocked."
        ),
        "self_care": [
            "Avoid knocking or scratching it, as these bleed easily",
            "If it bleeds, apply gentle pressure with a clean dressing",
            "Do not try to remove it yourself",
            "Photograph it to confirm it is not getting bigger",
        ],
        "red_flags": UNIVERSAL_RED_FLAGS
        + ["It bleeds heavily or repeatedly without being knocked"],
    },
}


# --- Level 1 groups (ADR 0007) ---------------------------------------------------
# Shown when the group model's answer is something other than a spot or growth.
# Same rules as above: no drug names or dosages; emergencies lead; guidance is
# conditional on a prediction that can be wrong. Urgency per group and the emergency
# signs below are general first-aid knowledge and need review by the project guide
# before anyone relies on them.

EMERGENCY = {
    "label": "Get emergency help now if any of these apply",
    "signs": [
        "Swelling of the face, lips, tongue or throat, or trouble breathing or swallowing",
        "Feeling faint, dizzy or confused, or a racing heartbeat",
        "A rash that does not fade when a clear glass is pressed against it",
        "A large burn, or any burn on the face, hands, feet or genitals",
        "Bleeding that does not stop after 10 minutes of firm pressure",
    ],
}

GROUP_GUIDANCE = {
    "rash": {
        "headline": "Usually not urgent -- keep an eye on it",
        "urgency": "routine",
        "plain": "A rash or patch of inflamed skin. Most are harmless and settle, but some need a doctor.",
        "what_it_is": (
            "Rashes have many causes: allergies, irritation from something that touched the "
            "skin, eczema-type conditions, heat, or a reaction to something eaten or taken. "
            "A photo alone often cannot tell them apart."
        ),
        "seek_care": (
            "See a doctor or pharmacist if it lasts more than a week or two, keeps coming back, "
            "is spreading, or is stopping you from sleeping."
        ),
        "self_care": [
            "Avoid anything new that touched the skin before it started (soap, cream, plant, fabric)",
            "Keep the area cool and avoid scratching",
            "Use plain, unperfumed products on the area",
            "Photograph it each day so you can show how it changes",
        ],
        "red_flags": [
            "It spreads quickly, blisters, or the skin starts to peel",
            "You have a fever or feel unwell with it",
            "It is around the eyes, mouth or genitals",
        ],
    },
    "infection": {
        "headline": "See a doctor or pharmacist in the next few days",
        "urgency": "prompt",
        "plain": "It may be a skin infection (fungal, bacterial or viral). These usually need treatment.",
        "what_it_is": (
            "Skin infections include fungal patches, infected hair follicles, cold sores, "
            "shingles, impetigo and others. Many spread to other skin or other people, and "
            "most clear faster with the right treatment."
        ),
        "seek_care": (
            "See a doctor or pharmacist in the next few days. Go sooner if it is spreading, "
            "painful, near the eye, or you have a fever."
        ),
        "self_care": [
            "Keep it clean and dry, and avoid touching or scratching it",
            "Wash your hands after touching it, and don't share towels",
            "Cover it loosely if it is weeping",
            "Note when it started and whether it is getting bigger",
        ],
        "red_flags": [
            "Redness spreading outward, or red streaks running from it",
            "Fever, chills, or feeling generally unwell",
            "Increasing pain, warmth or swelling, or pus",
            "Painful blisters in a band on one side of the body or face",
        ],
    },
    "bite_sting": {
        "headline": "Usually settles on its own -- watch for the warning signs",
        "urgency": "routine",
        "plain": "It looks like an insect bite or sting. Most settle on their own within days.",
        "what_it_is": (
            "Bites and stings often swell, itch and redden, and some people react with a much "
            "larger swelling than others. A big local reaction is common and usually not "
            "dangerous on its own."
        ),
        "seek_care": (
            "See a doctor or pharmacist if it is still getting bigger after two days, looks "
            "infected, or a ring-shaped rash appears after a tick bite."
        ),
        "self_care": [
            "Wash it with soap and water",
            "Hold something cold on it (wrapped, not directly on the skin) to ease swelling",
            "Raise the area if you can, and try not to scratch",
            "Remove a sting or tick carefully if one is still in the skin",
        ],
        "red_flags": [
            "Redness, warmth or swelling still spreading after 48 hours",
            "Fever, or feeling unwell after a bite",
            "A bullseye-shaped rash after a tick bite",
            "Many bites or stings at once",
        ],
    },
    "wound_scar": {
        "headline": "Usually heals on its own -- keep it clean and watch it",
        "urgency": "routine",
        "plain": "It looks like a wound, burn, bruise or scar -- an injury or what it left behind.",
        "what_it_is": (
            "Cuts, scrapes, burns and bruises usually heal well on their own when kept clean. "
            "Scars can stay raised, dark or pink for months while they mature, and that alone "
            "is not a sign of a problem."
        ),
        "seek_care": (
            "See a doctor for deep or gaping cuts, animal or human bites, dirty wounds (ask "
            "about tetanus protection), burns that blister or are larger than your palm, or a "
            "scar that keeps growing, hurts or changes."
        ),
        "self_care": [
            "Rinse a fresh wound with clean running water",
            "Press firmly with a clean cloth to stop bleeding, then cover it",
            "Cool a burn under cool running water for 20 minutes; do not use ice",
            "Keep it clean and check it daily for signs of infection",
        ],
        "red_flags": [
            "Increasing redness, warmth, swelling, pus or pain after the first day or two",
            "A scar or mark that grows, bleeds or changes colour",
            "Numbness or trouble moving a nearby finger or joint",
        ],
    },
}
