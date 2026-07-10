# Apptainer / Singularity

## Source of truth

Apptainer does not have an independent environment recipe. It converts an OCI
image built from `containers/Dockerfile.*` into a SIF deployment artifact.

```text
Dockerfile → OCI image or registry reference → Apptainer build → SIF
```

Example interface reserved by the build framework:

```bash
PHAL_OCI_URI=docker://registry.example/phal/md:m1-candidate1 \
containers/build.sh sif md
```

SIF files are immutable deployment artifacts, are ignored by Git, and should be
stored in a site image cache or artifact registry.

## HPC validation planned for Milestone 2

- NVIDIA GPU exposure with `--nv`;
- bind mounts for `/runtime` and site configuration;
- Slurm launch semantics;
- host driver / container CUDA compatibility;
- MPI ABI and process-launch compatibility.
