# PHAL Container Infrastructure

This directory is the canonical source for PHAL OCI images. Dockerfiles remain
the only image definitions; Apptainer SIF files are deployment artifacts and
must not be committed.

## Milestone status

The original `base`, `md`, `train`, and `cp2k` Dockerfiles remain unchanged
Milestone 1 scaffolds. They do not install scientific software and must not be
used as production images.

Milestone 2 adds one deliberately narrow target:

- `Dockerfile.train-probe`: an ARM64 probe of NVIDIA PyTorch, NequIP, Allegro,
  PHAL, Apptainer `--nv`, and one GH200 GPU.

It does not add LAMMPS, CP2K, a production training image, or multi-node MPI.
The candidate and its validation state are recorded in `versions.yaml`.

## Build and deployment flow

```text
Dockerfile.train-probe + repository root
                ↓ external Docker Buildx (linux/arm64)
        registry OCI image
                ↓ Miyabi Apptainer 1.3.5
          read-only SIF
                ↓ PBS job + --nv + /runtime bind
        GH200 validation results
```

The Docker build context is the repository root. The root `.dockerignore`
excludes mutable runtime data and build artifacts.

Inspect the resolved plan without Docker:

```bash
containers/build.sh plan train-probe
```

On an external ARM64-capable Buildx builder:

```bash
PHAL_REGISTRY=registry.example \
  containers/build.sh push train-probe
```

To export an OCI archive rather than push:

```bash
containers/build.sh oci train-probe
```

On Miyabi, a registry URI is mandatory; there is no Docker-daemon fallback:

```bash
module load apptainer/1.3.5
PHAL_OCI_URI=docker://registry.example/phal/train-probe:m2-train-probe-a \
  containers/build.sh sif train-probe
```

Mutable datasets, models, checkpoints, caches, and logs belong under the host
`runtime/` tree, mounted as `/runtime`. They are never copied into the image.

See `docs/container/milestone2-train-probe.md` for the complete build,
conversion, PBS submission, acceptance, and troubleshooting procedure.
