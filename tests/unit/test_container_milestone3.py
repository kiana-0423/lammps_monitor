"""Static Milestone 3 checks for the md-probe container."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONTAINERS = ROOT / "containers"
DOCKERFILE = CONTAINERS / "Dockerfile.md-probe"
BUILD_SCRIPT = CONTAINERS / "build.sh"
VERSIONS = CONTAINERS / "versions.yaml"
MODEL_SCRIPT = CONTAINERS / "scripts" / "build_md_probe_model.py"
VALIDATOR = CONTAINERS / "scripts" / "validate_md_stack.py"
STATIC_VALIDATOR = CONTAINERS / "scripts" / "validate_md_binary_static.py"
PAIR_PATCH = CONTAINERS / "scripts" / "patch_pair_nequip_allegro_lammps_10sep2025.py"
PROBE_DIR = CONTAINERS / "probes" / "md-probe"


def strings(value: Any):
    if isinstance(value, dict):
        for nested in value.values():
            yield from strings(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from strings(nested)
    elif isinstance(value, str):
        yield value


def test_md_probe_files_exist() -> None:
    expected = (
        DOCKERFILE,
        MODEL_SCRIPT,
        VALIDATOR,
        STATIC_VALIDATOR,
        PAIR_PATCH,
        PROBE_DIR / "README.md",
        PROBE_DIR / "data.h2",
        PROBE_DIR / "in.run0",
        PROBE_DIR / "in.run10",
    )
    assert all(path.is_file() for path in expected)


def test_md_probe_dockerfile_uses_fixed_sources_and_inherits_train_probe() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    lowered = text.lower()
    assert "arg base_image" in lowered
    assert "from ${base_image}" in lowered
    assert "ghcr.io/kiana-0423/train-probe" not in text
    assert "9792f6a9a32517780a8276c8ab201f17cae37d6b" in text
    assert "402b28390403aa92f34f14c2d5a9ff918acec598" in text
    assert "git clone --filter=blob:none" in text
    assert "git checkout --detach" in text
    assert "patch_pair_nequip_allegro_lammps_10sep2025.py" in text
    assert "main" not in lowered
    assert "latest" not in lowered
    assert "stable" not in lowered


def test_md_probe_cmake_contract_is_explicit() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    required = (
        "-D BUILD_MPI=on",
        "unset OPAL_PREFIX",
        "NVCC_WRAPPER_DEFAULT_COMPILER=/usr/bin/mpicxx.openmpi",
        "-D MPI_CXX_COMPILER=/usr/bin/mpicxx.openmpi",
        "-D CMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined",
        "ARG LAMMPS_BUILD_JOBS=4",
        '--parallel "${LAMMPS_BUILD_JOBS}"',
        "validate_md_binary_static.py",
        "-D BUILD_SHARED_LIBS=off",
        "-D PKG_KOKKOS=on",
        "-D Kokkos_ENABLE_CUDA=on",
        "-D Kokkos_ENABLE_SERIAL=on",
        "-D Kokkos_ENABLE_CUDA_LAMBDA=on",
        "-D Kokkos_ARCH_HOPPER90=on",
        "-D Kokkos_ARCH_ARMV9_GRACE=on",
        "-D KOKKOS_PREC=double",
        "-D NEQUIP_AOT_COMPILE=off",
        "torch/share/cmake",
        "torch._C._GLIBCXX_USE_CXX11_ABI",
    )
    combined = text + MODEL_SCRIPT.read_text(encoding="utf-8")
    for value in required:
        assert value in combined


def test_probe_model_is_marked_non_scientific() -> None:
    model_text = MODEL_SCRIPT.read_text(encoding="utf-8")
    readme = (PROBE_DIR / "README.md").read_text(encoding="utf-8")
    assert "integration_probe_only" in model_text
    assert "scientifically_valid" in model_text
    assert "training_steps" in model_text
    assert "integration_probe_only" in readme
    assert "scientifically_valid = false" in readme
    assert "not a formal MD simulation" in readme


def test_dockerfile_uses_static_binary_validation_on_local_builder() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert "lmp -h" not in text
    assert "grep -E \"allegro|allegro/kk\"" not in text
    assert "python /opt/phal/containers/scripts/validate_md_stack.py" not in text
    assert "validate_md_binary_static.py" in text
    assert 'LD_LIBRARY_PATH="/opt/hpcx/ucx/lib:/opt/hpcx/ucc/lib:${LD_LIBRARY_PATH:-}"' in text
    assert "import torch, nequip, allegro" in text
    assert "libcuda.so.1" not in text
    assert "ln -s /usr/local/cuda" not in text
    assert "Kokkos_ENABLE_CUDA=on" in text


def test_build_script_supports_md_probe_without_regressing_train_probe() -> None:
    script = BUILD_SCRIPT.read_text(encoding="utf-8")
    assert "md-probe" in script
    assert "candidates.md_stack_a" in script
    train_plan = subprocess.run(
        [str(BUILD_SCRIPT), "plan", "train-probe"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    md_plan = subprocess.run(
        [str(BUILD_SCRIPT), "plan", "md-probe"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    assert "target=train-probe" in train_plan
    assert "base_image=nvcr.io/nvidia/pytorch:25.09-py3@sha256:89172d8ef9c4641aacdcddf02c085bf7a736501b443ce2c4e0a660754f67106b" in train_plan
    assert "target=md-probe" in md_plan
    assert "platform=linux/arm64" in md_plan
    assert "base_image=ghcr.io/kiana-0423/train-probe@sha256:513aeaf7e997caa3b743ecc4b7c4d0defdfabd01a13de3f83841f2dc21810e86" in md_plan
    assert "lammps_commit=9792f6a9a32517780a8276c8ab201f17cae37d6b" in md_plan
    assert "pair_nequip_allegro_commit=402b28390403aa92f34f14c2d5a9ff918acec598" in md_plan


def test_versions_manifest_records_independent_md_candidate() -> None:
    data = yaml.safe_load(VERSIONS.read_text(encoding="utf-8"))
    candidate = data["candidates"]["md_stack_a"]
    assert data["candidates"]["train_stack_a"]["validation"]["miyabi_gpu"] is True
    assert candidate["image_tag"] == "m3-md-probe-a"
    assert candidate["base_train_probe_digest"] == "sha256:513aeaf7e997caa3b743ecc4b7c4d0defdfabd01a13de3f83841f2dc21810e86"
    assert candidate["lammps"]["commit"] == "9792f6a9a32517780a8276c8ab201f17cae37d6b"
    assert candidate["lammps"]["minimum_required"] == "10 Sep 2025"
    assert candidate["pair_nequip_allegro"]["commit"] == "402b28390403aa92f34f14c2d5a9ff918acec598"
    assert candidate["pair_nequip_allegro"]["compatibility_patch"] == (
        "containers/scripts/patch_pair_nequip_allegro_lammps_10sep2025.py"
    )
    assert candidate["build"]["kokkos"]["double_double"] is True
    assert candidate["build"]["mpi"] is True
    assert candidate["build"]["kokkos"]["cuda"] is True
    assert candidate["build"]["kokkos"]["serial"] is True
    assert candidate["build"]["kokkos"]["arch"]["gpu"] == "HOPPER90"
    assert candidate["probe_model"]["purpose"] == "integration_probe_only"
    assert candidate["probe_model"]["scientifically_valid"] is False
    assert candidate["validation"]["sif_conversion"] == "not_run"
    assert candidate["validation"]["miyabi_gh200"] == "not_run"

    floating = {"latest", "stable", "main"}
    assert not any(value.lower() in floating or value.lower().endswith(tuple(f":{x}" for x in floating)) for value in strings(candidate))


def test_validator_outputs_required_json_fields() -> None:
    text = VALIDATOR.read_text(encoding="utf-8")
    for field in (
        "platform.machine",
        "torch.cuda.is_available",
        "LAMMPS version",
        "pair_allegro_available",
        "compiled_model_sha256",
        "lmp_help_passed",
        "run0_passed",
        "run10_passed",
        "thermo_finite",
        "cpu_status",
        "cuda_status",
        "overall_status",
        "traceback",
    ):
        assert field in text


def test_static_validator_records_local_driver_limit_without_running_lmp() -> None:
    text = STATIC_VALIDATOR.read_text(encoding="utf-8")
    assert "not_available_on_local_builder" in text
    assert "libcuda.so.1 => not found" in text
    assert "pair_allegro_compiled" in text
    assert "lmp_runtime" in text
    assert "not_run" in text
    assert 'run(["ldd", str(LMP)])' in text
    assert 'run(["file", str(LMP)])' in text
    assert "lmp -h" not in text
