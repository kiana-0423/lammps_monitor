#!/usr/bin/env python3
"""Validate the ARM64 PyTorch/CUDA platform and emit one JSON document."""

from __future__ import annotations

import contextlib
import importlib
import json
import os
import platform
import sys
import traceback
from typing import Any


def _version(module: Any) -> str:
    return str(getattr(module, "__version__", "unknown"))


def main() -> int:
    require_cuda = os.environ.get("PHAL_REQUIRE_CUDA") == "1"
    result: dict[str, Any] = {
        "machine": platform.machine(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": None,
        "torch_cuda": None,
        "cuda_available": False,
        "cuda_device_count": 0,
        "gpu_name": None,
        "cuda_device_capability": None,
        "cuda_tensor_test": False,
        "nequip": None,
        "allegro": None,
        "phal_import": False,
        "require_cuda": require_cuda,
        "errors": [],
        "failures": [],
    }

    modules: dict[str, Any] = {}
    for name in ("torch", "nequip", "allegro", "hotspot_al"):
        try:
            # Some scientific packages print informational messages at import.
            # Keep stdout reserved for the machine-readable JSON document.
            with contextlib.redirect_stdout(sys.stderr):
                modules[name] = importlib.import_module(name)
        except Exception as exc:  # binary import failures must be reported too
            result["errors"].append(f"failed to import {name}: {type(exc).__name__}: {exc}")
            result["failures"].append(
                {
                    "stage": f"import-{name}",
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "traceback": traceback.format_exc(),
                    "selected_device": None,
                }
            )

    torch = modules.get("torch")
    if torch is not None:
        result["torch"] = _version(torch)
        result["torch_cuda"] = getattr(getattr(torch, "version", None), "cuda", None)
        try:
            result["cuda_available"] = bool(torch.cuda.is_available())
            result["cuda_device_count"] = int(torch.cuda.device_count())
            if result["cuda_available"]:
                result["gpu_name"] = str(torch.cuda.get_device_name(0))
                result["cuda_device_capability"] = list(torch.cuda.get_device_capability(0))
                left = torch.arange(16, dtype=torch.float32, device="cuda").reshape(4, 4)
                product = left @ left.transpose(0, 1)
                torch.cuda.synchronize()
                result["cuda_tensor_test"] = bool(torch.isfinite(product).all().item())
                if not result["cuda_tensor_test"]:
                    result["errors"].append("CUDA matrix multiplication produced NaN or Inf")
                if require_cuda and not any(token in result["gpu_name"].upper() for token in ("H100", "GH200")):
                    result["errors"].append(
                        f"expected a Miyabi-G H100/GH200 device, found {result['gpu_name']}"
                    )
            elif require_cuda:
                result["errors"].append("torch imported but CUDA is unavailable")
        except Exception as exc:
            result["errors"].append(f"CUDA tensor validation failed: {type(exc).__name__}: {exc}")
            result["failures"].append(
                {
                    "stage": "cuda-tensor",
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "traceback": traceback.format_exc(),
                    "selected_device": "cuda:0",
                }
            )

    if "nequip" in modules:
        result["nequip"] = _version(modules["nequip"])
    if "allegro" in modules:
        result["allegro"] = _version(modules["allegro"])
    result["phal_import"] = "hotspot_al" in modules
    result["package_versions"] = {name: _version(module) for name, module in modules.items()}
    for failure in result["failures"]:
        failure["package_versions"] = result["package_versions"]

    if result["machine"] not in {"aarch64", "arm64"}:
        result["errors"].append(f"expected ARM64 machine, found {result['machine']}")
    if sys.version_info < (3, 10):
        result["errors"].append("PHAL requires Python 3.10 or newer")

    cuda_passed = result["cuda_available"] and result["cuda_tensor_test"]
    passed = not result["errors"] and result["phal_import"] and all(
        name in modules for name in ("torch", "nequip", "allegro")
    )
    if require_cuda:
        passed = passed and cuda_passed
    result["status"] = "passed" if passed else "failed"
    print(json.dumps(result, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
