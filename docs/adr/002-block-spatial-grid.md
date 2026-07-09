# ADR 002: Block Mode Uses a Fixed Spatial Grid

## Status

Accepted.

## Context

Hotspot-centered spherical extraction can submit overlapping DFT jobs when many
nearby atoms trigger in consecutive frames. Dynamic clustering reduces some
duplicates, but the extracted region identity may drift over time.

## Decision

Block-aware PHAL maps anomalous atoms to stable spatial-grid block ids. A block
region is extracted as label core plus halo, buffers, and frozen boundary. PBC
neighboring blocks are merged when configured.

## Consequences

Stable ids make cooldown and bookkeeping straightforward. The tradeoff is that a
fixed grid may not follow molecular topology, so graph extraction and H capping
remain separate strategies for systems where connectivity matters more than
spatial locality.
