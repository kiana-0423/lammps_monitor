# Container Architecture

## Purpose

Define the long-lived boundary between PHAL source code, immutable execution
images, site configuration, and mutable runtime data.

## Milestone 1 decisions

- Dockerfiles are the only environment definitions.
- `containers/versions.yaml` is the only version manifest.
- Images are role-specific: base, MD, training, and CP2K labeling.
- OCI images are converted to SIF; Apptainer installation recipes are forbidden.
- Runtime data is mounted at `/runtime` and never stored in an image layer.
- No package or scientific software installation is part of this milestone.

## Planned sections

- image dependency graph;
- supply-chain metadata and SBOM;
- reproducible build and signing policy;
- CUDA/driver and MPI compatibility matrix;
- registry promotion across development, validation, and production.
