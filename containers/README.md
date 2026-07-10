# PHAL Container Infrastructure — Milestone 1

This directory is the canonical source for PHAL container images. Milestone 1
creates the build and filesystem contracts only. No Dockerfile installs Python,
CUDA, Torch, MLIP software, LAMMPS, CP2K, MPI, or any other package.

## Image roles

- `Dockerfile.base`: common operating-system foundation;
- `Dockerfile.md`: future GPU MLIP-MD runtime;
- `Dockerfile.train`: future GPU model-training runtime;
- `Dockerfile.cp2k`: future DFT labeling runtime.

Every Dockerfile receives `BASE_IMAGE` and `IMAGE_TAG` from `build.sh`. The
script resolves them from `versions.yaml`; Dockerfiles must not use `latest` or
define independent version pins.

Values set to `null` in `versions.yaml` are deliberately unresolved during
Milestone 1. Milestone 2 must resolve and validate a compatible version matrix
before adding installation layers.

## Docker, OCI, and Apptainer

Dockerfiles are the only environment definitions:

```text
versions.yaml
      ↓
Dockerfile.*
      ↓
Docker/BuildKit image (OCI)
      ├── registry push / OCI archive
      └── Apptainer conversion → SIF
```

Apptainer/Singularity must consume an OCI image produced from these
Dockerfiles. Do not add an Apptainer `%post` installation script or maintain a
second dependency definition. A SIF file is a deployment artifact and must
never be committed.

## Build interface

Inspect a resolved build without running Docker:

```bash
containers/build.sh plan base
containers/build.sh plan md
```

Future build/export/conversion entry points are already reserved:

```bash
containers/build.sh build base
containers/build.sh build-all
containers/build.sh tag md m1-candidate1
containers/build.sh oci md
containers/build.sh sif md
```

`sif` defaults to the explicitly tagged image in the local Docker daemon. CI or
HPC sites can supply an immutable registry reference:

```bash
PHAL_OCI_URI=docker://registry.example/phal/md:m1-candidate1 \
containers/build.sh sif md
```

Paths are derived from the script location. Registry namespaces, version files,
output directories, and OCI sources can be overridden through the environment;
no absolute workstation or cluster path is embedded in the build system.

## Runtime mounts

Images use `/runtime` as the container data root. The host `runtime/` directory
is mounted there for generated datasets, models, checkpoints, trajectories,
DFT tasks, logs, and caches. Source code and image layers must not contain these
artifacts.

See `docs/container/` for the architecture and deployment workflow.
