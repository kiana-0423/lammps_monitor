# ADR 004: Candidate Deduplication Uses Pair-Distance Histograms

## Status

Accepted.

## Context

The candidate pool needs a lightweight diversity filter before expensive DFT
labeling. Graph isomorphism and SOAP-like descriptors can be more expressive,
but they add dependencies and cost.

## Decision

The default fingerprint is a pair-distance histogram over extracted local
regions. A type-weighted variant is available for systems where atom identities
should influence deduplication.

## Consequences

The default is fast, dependency-light, and works well for coarse geometric
screening. It may miss stereochemical differences or topology-preserving
rearrangements, so richer descriptors can be added behind the same fingerprint
mode interface.
