# 4. Phone photos are the primary domain; dermoscopy is pretraining

Date: 2026-10-13

## Status
Accepted

## Context
The system is intended as a screening aid for ordinary people. Ordinary people photograph
a mole with the phone in their hand. They do not own a dermatoscope, and they do not shoot
on a high-end camera, transfer the file to a computer and upload it from there.

The obvious dataset, HAM10000, is entirely dermatoscopic: images taken through a magnifier
pressed against the skin, under controlled lighting. A model trained and evaluated on it
is being measured in a domain its users will never operate in. Published work in this area
largely ignores this and reports dermoscopy-only accuracy while claiming public screening
utility.

PAD-UFES-20 provides 2,298 real smartphone images of skin lesions with clinical metadata,
free and already present in the ISIC archive. It is, however, small — too small to train
a classifier on from scratch.

## Decision
Treat PAD-UFES-20 as the primary domain and HAM10000 as a pretraining corpus.

- Pretrain on HAM10000 (10,015 dermatoscopic images).
- Fine-tune on PAD-UFES-20.
- Report headline metrics on phone photos, with dermatoscopic numbers alongside for
  comparison.
- Separately measure and report the cold cross-domain result — HAM10000-trained,
  PAD-tested, no adaptation — as evidence of how much dermoscopy-only evaluation overstates
  real-world performance.

## Consequences
Headline accuracy will be lower than the numbers published papers report, because it is
measured in a harder and more honest setting. This is accepted and is the point.

Two pieces of real work follow and must be done explicitly rather than hand-waved:

1. **Label mapping.** PAD-UFES-20 has six classes (BCC, SCC, ACK, SEK, MEL, NEV) against
   HAM10000's seven. The mapping is documented, and non-overlapping classes are handled
   deliberately rather than silently dropped.
2. **Augmentation evaluation.** Phone-capture augmentation (blur, white balance, angle,
   JPEG artefacts) is a hypothesis, not a given. It is evaluated on whether it improves
   in-domain performance, and dropped if it does not.
