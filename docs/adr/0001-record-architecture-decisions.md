# 1. Record architecture decisions

Date: 2026-10-13

## Status
Accepted

## Context
This project runs for about two months with work happening in frequent small increments.
Decisions made in week one will be impossible to reconstruct by week seven, and a viva
will ask why things were done a particular way.

## Decision
Every significant decision gets a short dated file in `docs/adr/`: context, the decision,
and the consequences. Significant means it constrains later work or would be expensive to
reverse — the model architecture, the evaluation protocol, the API contract. Not library
version bumps.

## Consequences
The final report's methodology section can be assembled from these files rather than
reconstructed from memory.
