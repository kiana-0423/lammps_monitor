"""Static Milestone 2 checks that require no container runtime or GPU."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONTAINERS = ROOT / "containers"
DOCKERFILE = CONTAINERS / "Dockerfile.train-probe"
BUILD_SCRIPT = CONTAINERS / "build.sh"
PBS_SCRIPT = ROOT / "scripts" / "pbs" / "validate_train_probe.pbs"
VERSIONS = CONTAINERS / "versions.yaml"


def test_train_probe_files_exist() -> None:
    expected = (
        DOCKERFILE,
        ROOT / ".dockerignore",
        CONTAINERS / "scripts" / "validate_platform.py",
        CONTAINERS / "scripts" / "validate_train_stack.py",
        PBS_SCRIPT,
        ROOT / "docs" / "container" / "milestone2-train-probe.md",
    )
    assert all(path.is_file() for path in expected)


def test_train_probe_dockerfile_is_pinned_and_inherits_cuda_torch() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    lowered = text.lower()
    assert "arg base_image" in lowered
    assert "from ${base_image}" in lowered
    assert "latest" not in lowered
    assert "pip install torch" not in lowered
    assert "apt-get install cuda" not in lowered
    assert "nvidia/cuda" not in lowered
    assert 'copy src ./src' in lowered
    assert "python -m pip install -e ." in lowered
    for path in ("/runtime/models", "/runtime/datasets", "/runtime/checkpoints", "/runtime/logs"):
        assert path in text


def test_build_script_uses_root_context_and_arm64_default() -> None:
    text = BUILD_SCRIPT.read_text(encoding="utf-8")
    assert '"${PROJECT_ROOT}"' in text
    assert "Dockerfile.train-probe" not in text  # target resolution stays generic
    assert "PHAL_PLATFORM" in text
    assert "PHAL_BASE_IMAGE" in text
    assert "PHAL_IMAGE_TAG" in text
    assert "PHAL_REGISTRY" in text
    assert "PHAL_OCI_URI" in text
    assert "--metadata-file" in text

    result = subprocess.run(
        [str(BUILD_SCRIPT), "plan", "train-probe"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    assert "platform=linux/arm64" in result.stdout
    assert f"build_context={ROOT}" in result.stdout
    assert (
        "base_image=nvcr.io/nvidia/pytorch:25.09-py3"
        "@sha256:89172d8ef9c4641aacdcddf02c085bf7a736501b443ce2c4e0a660754f67106b"
    ) in result.stdout


def test_sif_requires_registry_uri_before_apptainer() -> None:
    env = os.environ.copy()
    env.pop("PHAL_OCI_URI", None)
    result = subprocess.run(
        [str(BUILD_SCRIPT), "sif", "train-probe"],
        cwd=ROOT,
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2
    assert "PHAL_OCI_URI is required" in result.stderr
    assert "Docker daemon" in result.stderr


def test_versions_manifest_has_planned_candidate_and_no_floating_refs() -> None:
    data = yaml.safe_load(VERSIONS.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    assert data["platform"]["architecture"] == "arm64"
    assert data["platform"]["apptainer"] == "1.3.5"
    candidate = data["candidates"]["train_stack_a"]
    assert candidate["base_image"] == "nvcr.io/nvidia/pytorch:25.09-py3"
    assert candidate["base_image_digest"] == "sha256:89172d8ef9c4641aacdcddf02c085bf7a736501b443ce2c4e0a660754f67106b"
    assert candidate["explicit"]["nequip"] == "0.18.0"
    assert candidate["explicit"]["allegro"] == "0.8.3"
    assert candidate["status"] == "planned"
    assert data["validated"]["selected_train_stack"] is None

    def strings(value: Any):
        if isinstance(value, dict):
            for nested in value.values():
                yield from strings(nested)
        elif isinstance(value, list):
            for nested in value:
                yield from strings(nested)
        elif isinstance(value, str):
            yield value

    floating = {"latest", "stable", "main"}
    assert not any(value.lower() in floating or value.lower().endswith(tuple(f":{x}" for x in floating)) for value in strings(data))


def test_pbs_script_has_miyabi_gpu_contract() -> None:
    text = PBS_SCRIPT.read_text(encoding="utf-8")
    assert 'cd "$PBS_O_WORKDIR"' in text
    assert "module purge" in text
    assert "module load apptainer/1.3.5" in text
    assert text.count("apptainer exec") == 3
    assert text.count("--cleanenv") == 3
    assert text.count("--nv") == 3
    assert '--bind "$PBS_O_WORKDIR/runtime:/runtime"' in text
    assert "APPTAINER_TMPDIR" in text
    assert "/logs/validation" in text
    assert "#PBS -q debug-g" in text
    assert "#PBS -W group_list=gw34" in text
    assert "gpu-smoke-${PBS_JOBID}.json.log" in text
    assert "APPTAINERENV_PHAL_REQUIRE_CUDA=1" in text
    assert "<MIYABI_" not in text


def test_validators_record_real_gpu_and_inference_evidence() -> None:
    platform_validator = (CONTAINERS / "scripts" / "validate_platform.py").read_text(encoding="utf-8")
    train_validator = (CONTAINERS / "scripts" / "validate_train_stack.py").read_text(encoding="utf-8")
    assert "PHAL_REQUIRE_CUDA" in platform_validator
    assert "get_device_capability" in platform_validator
    assert "device=\"cuda\"" in platform_validator
    assert "cuda_tensor_test" in platform_validator
    assert "traceback.format_exc()" in platform_validator
    assert "TOTAL_ENERGY_KEY" in train_validator
    assert "FORCE_KEY" in train_validator
    assert "energy_finite" in train_validator
    assert "forces_finite" in train_validator
    assert "force_gradient_computed" in train_validator
    assert "allegro-cuda-inference" in train_validator
