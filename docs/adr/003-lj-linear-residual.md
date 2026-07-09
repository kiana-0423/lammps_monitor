# ADR 003: LJ Residual Uses a Linear 2x2 Solve

## Status

Accepted.

## Context

The LJ residual monitor is a physics-aware screen for local force mismatch. It
needs to run inside online monitoring, so nonlinear optimization per atom would
be too expensive and brittle for large systems.

## Decision

PHAL projects local forces onto a two-parameter Lennard-Jones force basis and
solves the resulting 2x2 linear system. The fast path rejects singular or
ill-conditioned Gram matrices before accepting a fit.

## Consequences

The method is cheap and deterministic, suitable for staged OOD scoring. It is a
screening signal rather than a chemically complete force model, so invalid fits
are treated conservatively.
