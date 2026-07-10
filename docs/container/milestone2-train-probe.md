# Milestone 2: Miyabi-G ARM64 training probe

## Goal and scope

This milestone tests one dependency chain before PHAL commits to production
images:

```text
linux/arm64 → NVIDIA ARM SBSA PyTorch → Python/CUDA/Torch
→ NequIP → Allegro → PHAL → Apptainer --nv → GH200
→ finite CUDA tensors and finite Allegro energy/forces
```

The train stack is first because it validates the highest-risk intersection:
ARM64 binary availability, the site GPU driver, NVIDIA user-space CUDA, PyTorch,
and rapidly evolving NequIP/Allegro APIs. Implementing LAMMPS and CP2K at the
same time would make failures difficult to isolate. The Milestone 1 MD, train,
and CP2K Dockerfiles therefore remain scaffolds.

## Candidate stack

`containers/versions.yaml` records candidate `train_stack_a`:

- base: `nvcr.io/nvidia/pytorch:25.09-py3`;
- OS, Python, CUDA, Torch: inherited from the base image;
- NequIP: `0.18.0`, explicitly installed;
- PyPI distribution `nequip-allegro`: `0.8.3`, explicitly installed;
- PHAL: installed from this repository.

The NGC tag is fixed and has an ARM64 manifest. It uses CUDA 13.0.1 and is a
closer candidate for Miyabi's NVIDIA 25.9 / CUDA 13.0 environment than CUDA
13.1 or 13.2 images. A real NGC Registry V2 query on 2026-07-10 fixed the ARM64
manifest digest to
`sha256:89172d8ef9c4641aacdcddf02c085bf7a736501b443ce2c4e0a660754f67106b`.
The sanitized manifest evidence is stored under `docs/container/evidence/`.
Every runtime validation flag remains false until its corresponding real
operation passes.

## External builder requirements

The build machine needs:

- Docker with Buildx/BuildKit;
- network access to `nvcr.io`, PyPI, and the destination registry;
- credentials required by those registries;
- enough local space for the roughly 10 GB compressed NGC base and build
  layers;
- a Buildx worker capable of `linux/arm64`, either native ARM64 or correctly
  configured emulation.

It does not need a GPU because no build step calls CUDA.

Confirm that the base image advertises `linux/arm64` before building:

```bash
docker buildx imagetools inspect nvcr.io/nvidia/pytorch:25.09-py3
```

The manifest output must contain `Platform: linux/arm64` and agree with the
recorded digest. `build.sh` automatically combines the candidate tag and digest
into an immutable `tag@sha256:...` build argument.

## Build and push OCI

Review the resolved inputs:

```bash
containers/build.sh plan train-probe
```

The exact Buildx form is:

```bash
docker buildx build \
  --platform linux/arm64 \
  --file containers/Dockerfile.train-probe \
  --build-arg BASE_IMAGE=nvcr.io/nvidia/pytorch:25.09-py3@sha256:89172d8ef9c4641aacdcddf02c085bf7a736501b443ce2c4e0a660754f67106b \
  --build-arg IMAGE_TAG=m2-train-probe-a \
  --build-arg NEQUIP_VERSION=0.18.0 \
  --build-arg ALLEGRO_VERSION=0.8.3 \
  --tag registry.example/phal/train-probe:m2-train-probe-a \
  --metadata-file build/evidence/train-probe-build-metadata.json \
  --push \
  . 2>&1 | tee build/evidence/train-probe-build.log
```

The repository wrapper produces the same build from the manifest:

```bash
PHAL_PLATFORM=linux/arm64 \
PHAL_REGISTRY=registry.example \
PHAL_IMAGE_TAG=m2-train-probe-a \
  containers/build.sh push train-probe
```

Do not install another CUDA toolkit or a PyPI Torch wheel in this image. CUDA
and Torch are properties of the NGC base candidate.

For CPU-only OCI validation, leave `PHAL_REQUIRE_CUDA` unset; imports and the
CPU Allegro inference remain mandatory while CUDA is reported as unavailable.
The PBS script exports `PHAL_REQUIRE_CUDA=1`, making the GH200 CUDA tensor and
GPU Allegro inference mandatory on Miyabi-G.

## Convert the registry image to SIF on Miyabi

Run this on a Miyabi login node; it converts an OCI image and does not perform
GPU validation:

```bash
cd /work/gw34/w34000/project/lammps_monitor
module purge
module load apptainer/1.3.5

export APPTAINER_CACHEDIR=/work/gw34/w34000/.cache/apptainer
export APPTAINER_TMPDIR=/tmp/$USER/phal-apptainer-build
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

PHAL_OCI_URI=docker://registry.example/phal/train-probe:m2-train-probe-a \
  containers/build.sh sif train-probe
```

The expected output path is:

```text
build/containers/phal-train-probe-m2-train-probe-a-linux-arm64.sif
```

For a private registry, authenticate with `apptainer registry login` or a
site-approved auth file. Never commit credentials or tokens.

## Submit the GH200 validation

Edit the two obvious placeholders at the top of
`scripts/pbs/validate_train_probe.pbs`:

- `<MIYABI_G_QUEUE>`: an authorized Miyabi-G queue;
- `<MIYABI_PROJECT_GROUP>`: the user's authorized project group.

Then submit the SIF path through PBS:

```bash
IMAGE="$PWD/build/containers/phal-train-probe-m2-train-probe-a-linux-arm64.sif"
qsub -v PHAL_TRAIN_PROBE_IMAGE="$IMAGE" scripts/pbs/validate_train_probe.pbs
```

Inspect the job and results:

```bash
qstat
find runtime/logs/validation -maxdepth 1 -type f -print
cat runtime/logs/validation/platform-<JOB_ID>.json.log
cat runtime/logs/validation/train-stack-<JOB_ID>.json.log
```

Do not run the CUDA validators on a login node. The PBS script uses
`apptainer exec --cleanenv --nv`, binds the repository `runtime/` directory to
`/runtime`, and forces CUDA inference to be required.

## Acceptance criteria

The probe passes only when both scripts exit zero on a Miyabi-G compute node:

1. machine is `aarch64` or `arm64`;
2. Python is at least 3.10;
3. Torch, NequIP, Allegro, and PHAL import;
4. Torch reports CUDA available and at least one device;
5. a real CUDA matrix multiplication returns only finite values;
6. a randomly initialized tiny Allegro model produces finite energy and forces
   on CPU;
7. the same model test produces finite energy and forces on GH200 CUDA.

The tiny model checks software execution only. Its random outputs have no
scientific meaning and do not validate a trained model or deployment format.

After evidence is reviewed, update only the corresponding validation booleans
in `versions.yaml`. Set `validated.selected_train_stack` only after the image
digest, SIF conversion, GH200 test, and Allegro CPU/GPU inference all pass.

## Common failures

- **Image is amd64:** the Buildx command omitted `--platform linux/arm64`, or
  the wrong manifest was selected. Inspect both the base and pushed image.
- **No ARM64 base manifest:** select a documented NVIDIA ARM SBSA tag; do not
  emulate the runtime image as amd64 on Miyabi.
- **Apptainer omitted `--nv`:** host GPU devices and driver libraries will not
  be injected into the container.
- **Torch imports but CUDA is false:** verify the job is on Miyabi-G, `--nv` is
  present, and the base CUDA is compatible with the host driver.
- **No GPU on login node:** expected. Submit the PBS validation job.
- **`/tmp` fills:** point `APPTAINER_TMPDIR` at a sufficiently large local
  temporary location and remove abandoned build directories according to site
  policy.
- **Registry authentication fails:** log in to NGC/destination registry using
  approved credentials; do not put credentials in Dockerfiles or PBS scripts.
- **Python is too old:** PHAL, NequIP 0.18, and Allegro 0.8 require Python 3.10
  or newer. Use the inherited container Python, not the host Python 3.9.
- **Allegro imports but inference fails:** imports validate packaging only.
  Preserve the `blocked` result and its exception; do not mark inference true.
- **Pip constraint conflict:** NGC images contain `/etc/pip/constraint.txt`.
  Investigate the conflicting dependency instead of replacing the inherited
  Torch/CUDA stack silently.
