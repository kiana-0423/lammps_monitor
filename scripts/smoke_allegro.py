#!/usr/bin/env python
"""Lightweight Allegro/NequIP runtime smoke test."""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
from importlib import metadata
from typing import Any


EXIT_OK = 0
EXIT_RUNTIME_ERROR = 1
EXIT_IMPORT_ERROR = 2
EXIT_DEVICE_UNAVAILABLE = 3
EXIT_ENV_ERROR = 4

SUPPORTED_DEVICES = {"auto", "cpu", "cuda", "mps"}


def _version(distribution: str, module: Any | None = None) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return getattr(module, "__version__", "unknown") if module is not None else "unknown"


def _import_required(module_name: str) -> tuple[Any | None, str | None]:
    try:
        return importlib.import_module(module_name), None
    except Exception as exc:  # pragma: no cover - exercised by real-env smoke test
        return None, f"{type(exc).__name__}: {exc}"


def _print_cli_version(command: str) -> None:
    path = shutil.which(command)
    if path is None:
        print(f"{command} CLI: not found")
        return

    print(f"{command} CLI: {path}")
    try:
        result = subprocess.run(
            [path, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as exc:
        print(f"{command} --version: unavailable ({type(exc).__name__}: {exc})")
        return

    output = (result.stdout or result.stderr).strip()
    if output:
        print(f"{command} --version: {output.splitlines()[0]}")
    else:
        print(f"{command} --version: exited {result.returncode} with no output")


def _select_device(
    requested_device: str,
    *,
    cuda_available: bool,
    mps_available: bool,
) -> tuple[str, int | None]:
    if requested_device not in SUPPORTED_DEVICES:
        print(f"error: unsupported ALLEGRO_SMOKE_DEVICE={requested_device!r}")
        return "", EXIT_DEVICE_UNAVAILABLE
    if requested_device == "cuda" and not cuda_available:
        print("selected device: cuda")
        print("error: ALLEGRO_SMOKE_DEVICE=cuda was requested, but CUDA is unavailable")
        return "", EXIT_DEVICE_UNAVAILABLE
    if requested_device == "mps" and not mps_available:
        print("selected device: mps")
        print("error: ALLEGRO_SMOKE_DEVICE=mps was requested, but MPS is unavailable")
        return "", EXIT_DEVICE_UNAVAILABLE
    if requested_device == "auto":
        if cuda_available:
            return "cuda", None
        if mps_available:
            return "mps", None
        return "cpu", None
    return requested_device, None


def main() -> int:
    try:
        print(f"Python version: {sys.version.split()[0]}")
        print(f"Python executable: {sys.executable}")
        conda_env = os.environ.get("CONDA_DEFAULT_ENV", "<unset>")
        print(f"Conda environment: {conda_env}")
        required_env = os.environ.get("ALLEGRO_REQUIRED_CONDA_ENV")
        if required_env and conda_env != required_env:
            print(f"error: this smoke test must run inside conda environment {required_env}")
            return EXIT_ENV_ERROR

        torch, torch_error = _import_required("torch")
        if torch is None:
            print(f"torch import status: failed ({torch_error})")
            return EXIT_IMPORT_ERROR

        print(f"torch version: {getattr(torch, '__version__', 'unknown')}")
        cuda_available = bool(torch.cuda.is_available())
        mps_available = bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
        print(f"cuda availability: {cuda_available}")
        print(f"mps availability: {mps_available}")

        requested_device = os.environ.get("ALLEGRO_SMOKE_DEVICE", "auto").lower()
        selected_device, device_error = _select_device(
            requested_device,
            cuda_available=cuda_available,
            mps_available=mps_available,
        )
        if device_error is not None:
            return device_error
        print(f"selected device: {selected_device}")

        nequip, nequip_error = _import_required("nequip")
        allegro, allegro_error = _import_required("allegro")
        ase, ase_error = _import_required("ase")
        print(
            "nequip import status: "
            + (f"ok ({_version('nequip', nequip)})" if nequip is not None else f"failed ({nequip_error})")
        )
        print(
            "allegro import status: "
            + (f"ok ({_version('nequip-allegro', allegro)})" if allegro is not None else f"failed ({allegro_error})")
        )
        print(
            "ase import status: "
            + (f"ok ({_version('ase', ase)})" if ase is not None else f"failed ({ase_error})")
        )
        if nequip is None or allegro is None or ase is None:
            return EXIT_IMPORT_ERROR

        cpu_tensor = torch.tensor([1.0, 2.0, 3.0], device="cpu")
        cpu_result = (cpu_tensor * 2.0).sum().item()
        print(f"cpu tensor sanity: ok ({cpu_result:.1f})")

        if selected_device != "cpu":
            device_tensor = torch.tensor([1.0, 2.0, 3.0], device=selected_device)
            device_result = (device_tensor + 1.0).sum().item()
            print(f"{selected_device} tensor sanity: ok ({device_result:.1f})")

        _print_cli_version("nequip-train")
        _print_cli_version("allegro-train")
        return EXIT_OK
    except Exception as exc:  # pragma: no cover - defensive real-env error path
        print(f"runtime error: {type(exc).__name__}: {exc}")
        return EXIT_RUNTIME_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
