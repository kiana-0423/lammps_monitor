#!/usr/bin/env python3
"""Run a real, randomly initialized tiny Allegro forward/force calculation."""

from __future__ import annotations

import contextlib
import importlib
import json
import os
import sys
import traceback
from typing import Any


def _version(module: Any) -> str:
    return str(getattr(module, "__version__", "unknown"))


def _failure(stage: str, exc: Exception, device: str | None, versions: dict[str, str]) -> dict[str, Any]:
    return {
        "stage": stage,
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
        "traceback": traceback.format_exc(),
        "selected_device": device,
        "package_versions": versions,
    }


def _run_inference(torch: Any, device: str) -> dict[str, Any]:
    """Build a minimal Allegro model and evaluate finite energy and forces."""

    from allegro.model import AllegroModel
    from nequip.data import AtomicDataDict
    from nequip.utils.global_state import set_global_state

    set_global_state(allow_tf32=False)
    model = AllegroModel(
        seed=123,
        model_dtype="float32",
        type_names=["H"],
        r_max=3.0,
        l_max=1,
        parity=True,
        radial_chemical_embed={
            "_target_": "allegro.nn.TwoBodyBesselScalarEmbed",
            "num_bessels": 4,
            "bessel_trainable": False,
            "polynomial_cutoff_p": 6,
        },
        num_layers=1,
        num_scalar_features=8,
        num_tensor_features=4,
        scalar_embed_mlp_hidden_layers_width=8,
        allegro_mlp_hidden_layers_width=8,
        readout_mlp_hidden_layers_width=8,
        avg_num_neighbors=1.0,
        per_type_energy_scales=1.0,
        per_type_energy_shifts=0.0,
        do_derivatives=True,
        compile_mode="eager",
    ).to(device)
    model.eval()

    data = {
        AtomicDataDict.POSITIONS_KEY: torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]], dtype=torch.float32, device=device
        ),
        AtomicDataDict.EDGE_INDEX_KEY: torch.tensor(
            [[0, 1], [1, 0]], dtype=torch.long, device=device
        ),
        AtomicDataDict.ATOM_TYPE_KEY: torch.zeros(2, dtype=torch.long, device=device),
    }

    # ForceStressOutput needs autograd even though this is inference.
    with torch.enable_grad():
        output = model(data)
    energy = output[AtomicDataDict.TOTAL_ENERGY_KEY]
    forces = output[AtomicDataDict.FORCE_KEY]
    energy_finite = bool(torch.isfinite(energy).all().item())
    forces_finite = bool(torch.isfinite(forces).all().item())
    finite = energy_finite and forces_finite
    if device == "cuda":
        torch.cuda.synchronize()

    return {
        "status": "passed" if finite else "failed",
        "device": device,
        "randomly_initialized_model": True,
        "energy_shape": list(energy.shape),
        "force_shape": list(forces.shape),
        "energy_finite": energy_finite,
        "forces_finite": forces_finite,
        "finite_energy_and_forces": finite,
        "force_gradient_computed": True,
    }


def main() -> int:
    result: dict[str, Any] = {
        "status": "blocked",
        "imports": {},
        "api": {"status": "blocked", "detail": None},
        "inference": {
            "cpu": {"status": "blocked"},
            "cuda": {"status": "blocked"},
        },
        "errors": [],
        "failures": [],
    }

    modules: dict[str, Any] = {}
    for name in ("torch", "nequip", "allegro", "hotspot_al"):
        try:
            with contextlib.redirect_stdout(sys.stderr):
                module = importlib.import_module(name)
            modules[name] = module
            result["imports"][name] = {"status": "passed", "version": _version(module)}
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            result["imports"][name] = {"status": "failed", "detail": detail}
            result["errors"].append(f"failed to import {name}: {detail}")
            result["failures"].append(_failure(f"import-{name}", exc, None, {}))

    versions = {name: _version(module) for name, module in modules.items()}
    for failure in result["failures"]:
        failure["package_versions"] = versions

    if len(modules) != 4:
        result["api"]["detail"] = "required imports failed; model API was not inspected"
        print(json.dumps(result, sort_keys=True))
        return 1

    try:
        from allegro.model import AllegroModel

        if not callable(AllegroModel):
            raise TypeError("allegro.model.AllegroModel is not callable")
        result["api"] = {
            "status": "passed",
            "detail": "AllegroModel builder is available; a tiny eager model will be constructed",
        }
    except Exception as exc:
        detail = f"Allegro model API unavailable: {type(exc).__name__}: {exc}"
        result["api"] = {"status": "blocked", "detail": detail}
        result["errors"].append(detail)
        result["failures"].append(_failure("allegro-model-api", exc, None, versions))
        print(json.dumps(result, sort_keys=True))
        return 1

    torch = modules["torch"]
    try:
        with contextlib.redirect_stdout(sys.stderr):
            result["inference"]["cpu"] = _run_inference(torch, "cpu")
        if result["inference"]["cpu"]["status"] != "passed":
            result["errors"].append("CPU Allegro inference produced non-finite energy or forces")
            result["failures"].append(
                {
                    "stage": "allegro-cpu-inference",
                    "exception_type": "NonFiniteOutput",
                    "exception_message": "energy or forces contain NaN/Inf",
                    "traceback": None,
                    "selected_device": "cpu",
                    "package_versions": versions,
                }
            )
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        result["inference"]["cpu"] = {"status": "blocked", "detail": detail}
        result["errors"].append(f"CPU Allegro inference blocked: {detail}")
        result["failures"].append(_failure("allegro-cpu-inference", exc, "cpu", versions))

    cuda_available = bool(torch.cuda.is_available())
    if cuda_available:
        try:
            with contextlib.redirect_stdout(sys.stderr):
                result["inference"]["cuda"] = _run_inference(torch, "cuda")
            if result["inference"]["cuda"]["status"] != "passed":
                result["errors"].append("CUDA Allegro inference produced non-finite energy or forces")
                result["failures"].append(
                    {
                        "stage": "allegro-cuda-inference",
                        "exception_type": "NonFiniteOutput",
                        "exception_message": "energy or forces contain NaN/Inf",
                        "traceback": None,
                        "selected_device": "cuda",
                        "package_versions": versions,
                    }
                )
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            result["inference"]["cuda"] = {"status": "blocked", "detail": detail}
            result["errors"].append(f"CUDA Allegro inference blocked: {detail}")
            result["failures"].append(_failure("allegro-cuda-inference", exc, "cuda", versions))
    else:
        result["inference"]["cuda"] = {
            "status": "skipped",
            "detail": "torch.cuda.is_available() is false",
        }
        if os.environ.get("PHAL_REQUIRE_CUDA") == "1":
            result["errors"].append("CUDA Allegro inference is required but CUDA is unavailable")

    required_statuses = [result["api"]["status"], result["inference"]["cpu"]["status"]]
    if cuda_available or os.environ.get("PHAL_REQUIRE_CUDA") == "1":
        required_statuses.append(result["inference"]["cuda"]["status"])
    passed = not result["errors"] and all(status == "passed" for status in required_statuses)
    result["status"] = "passed" if passed else "blocked"
    print(json.dumps(result, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
