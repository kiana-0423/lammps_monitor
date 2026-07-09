# ADR 001: Atom-Wise OOD Instead of Frame-Level Selection

## Status

Accepted.

## Context

Large reactive MLIP-MD trajectories can contain localized failures while the
rest of the frame remains well described by the model. Frame-level uncertainty
selection treats the whole structure as equally informative and tends to push
large DFT labeling jobs downstream.

## Decision

PHAL uses atom-wise OOD scores as the primary trigger surface. Frame-level events
are still recorded, but extraction and masking are driven by anomalous atoms and
their local context.

## Consequences

This reduces labeling cost and makes trigger reasons interpretable at the atom
level. It also requires careful hotspot merging, buffer handling, and per-atom
training masks to avoid overfitting truncated regions.
