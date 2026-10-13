# 3. Build around the deployment gap, not around novelty

Date: 2026-10-13

## Status
Accepted

## Context
A literature check found the intended project — HAM10000 + CNN + Grad-CAM + web app +
treatment suggestions — already published many times, including a paper the team's own
draft cites as reference [10] (an explainable lightweight CNN on HAM10000 with Grad-CAM
and Grad-CAM++, deployed for real-time use). Care/remedy recommendation on top of lesion
classification is also published. The lightweight-architecture comparison is standard
practice rather than a contribution.

Separately, the project has no budget: no paid compute, hosting or subscriptions, and
publication is not a goal. The goal is a strong final-year project and portfolio piece.

## Decision
Stop pursuing novelty. Build around a real, honest weakness instead: published systems
train and test on dermatoscopic images while claiming to screen the general public, who
photograph lesions with phones. Measure that gap on PAD-UFES-20, mitigate part of it, and
make the system abstain when the input is outside what it handles. Add measured Grad-CAM
faithfulness as a second strand.

Rejected alternatives: chasing a novel architecture (not achievable in two months without
compute), and shipping the standard pipeline unchanged (indistinguishable from a tutorial).

## Consequences
The headline result may be a poor cross-domain number. That is accepted and will be
reported — it is the most interesting thing the project will have to say, and the
abstention mechanism exists precisely because of it. Scope stays within free tiers:
Colab for training, Hugging Face Spaces for hosting, GitHub Actions for CI.
