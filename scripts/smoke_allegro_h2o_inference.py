#!/usr/bin/env python
"""H2O Allegro/NequIP model loading and inference smoke test."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys
from typing import Any


EXIT_OK = 0
EXIT_RUNTIME_ERROR = 1
EXIT_IMPORT_ERROR = 2
EXIT_DEVICE_UNAVAILABLE = 3
EXIT_MODEL_PATH_ERROR = 4
EXIT_MODEL_LOAD_FAILED = 5
EXIT_INFERENCE_FAILED = 6
EXIT_ELEMENT_MISMATCH = 7

REQUIRED_SYMBOLS = {"H", "O"}


def _import_required(module_name: str) -> tuple[Any | None, str | None]:
    try:
        return importlib.import_module(module_name), None
    except Exception as exc:  # pragma: no cover - real environment path
        return None, f"{type(exc).__name__}: {exc}"


def _select_device(torch: Any) -> tuple[str | None, int | None]:
    requested = os.environ.get("ALLEGRO_SMOKE_DEVICE", "cpu").lower()
    mps_available = bool(
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )
    print(f"requested device: {requested}")
    print(f"cuda availability: {bool(torch.cuda.is_available())}")
    print(f"mps availability: {mps_available}")

    if requested == "cuda":
        print("error: 当前是 Mac Allegro smoke test，不支持 CUDA。")
        return None, EXIT_DEVICE_UNAVAILABLE
    if requested not in {"auto", "cpu", "mps"}:
        print(f"error: unsupported ALLEGRO_SMOKE_DEVICE={requested!r}")
        return None, EXIT_DEVICE_UNAVAILABLE
    if requested == "mps" and not mps_available:
        print("error: ALLEGRO_SMOKE_DEVICE=mps was requested, but MPS is unavailable")
        return None, EXIT_DEVICE_UNAVAILABLE
    return ("mps" if requested == "mps" else "cpu"), None


def _build_h2o(ase_module: Any) -> Any:
    atoms = ase_module.Atoms(
        "OH2",
        positions=[
            [0.000000, 0.000000, 0.000000],
            [0.958400, 0.000000, 0.000000],
            [-0.239000, 0.927000, 0.000000],
        ],
    )
    atoms.pbc = False
    return atoms


def _calculator_class() -> tuple[Any | None, str | None]:
    errors: list[str] = []
    for module_name in ("nequip.integrations.ase", "nequip.ase"):
        module, error = _import_required(module_name)
        if module is None:
            errors.append(f"{module_name}: {error}")
            continue
        calculator = getattr(module, "NequIPCalculator", None)
        if calculator is not None:
            print(f"NequIPCalculator import: ok ({module_name}.NequIPCalculator)")
            return calculator, None
        errors.append(f"{module_name}: NequIPCalculator not found")
    return None, "; ".join(errors)


def _metadata_symbols(metadata: Any) -> set[str]:
    symbols: set[str] = set()
    if not isinstance(metadata, dict):
        return symbols
    candidate_keys = (
        "chemical_symbols",
        "chemical_species",
        "allowed_species",
        "type_names",
        "atom_type_names",
    )
    for key in candidate_keys:
        value = metadata.get(key)
        if isinstance(value, str):
            symbols.add(value)
        elif isinstance(value, (list, tuple, set)):
            symbols.update(str(item) for item in value)
    mapper = metadata.get("chemical_species_to_atom_type_map")
    if isinstance(mapper, dict):
        symbols.update(str(key) for key in mapper)
    return symbols


def _looks_like_element_mismatch(message: str) -> bool:
    lowered = message.lower()
    markers = (
        "chemical species",
        "atom type",
        "type mapper",
        "species",
        "unknown element",
        "keyerror",
    )
    return any(marker in lowered for marker in markers) and (
        " h" in lowered or " o" in lowered or "'h'" in lowered or "'o'" in lowered
    )


def _try_load_calculator(calculator_cls: Any, model_path: Path, device: str) -> tuple[Any | None, str | None, bool]:
    load_errors: list[str] = []

    from_compiled = getattr(calculator_cls, "from_compiled_model", None)
    if from_compiled is not None:
        for mapper in (True, None):
            try:
                calculator = from_compiled(
                    model_path,
                    device=device,
                    chemical_species_to_atom_type_map=mapper,
                )
                print(
                    "model load status: ok "
                    f"(NequIPCalculator.from_compiled_model, map={mapper})"
                )
                return calculator, None, False
            except Exception as exc:
                message = f"from_compiled_model(map={mapper}): {type(exc).__name__}: {exc}"
                load_errors.append(message)
                if _looks_like_element_mismatch(message):
                    return None, message, True

    try:
        from nequip.data.transforms import ChemicalSpeciesToAtomTypeMapper, NeighborListTransform
        from nequip.model.saved_models.checkpoint import ModelFromCheckpoint
        from nequip.utils.global_state import set_global_state

        set_global_state()
        model = ModelFromCheckpoint(str(model_path))
        if hasattr(model, "keys") and "sole_model" in model:
            model = model["sole_model"]
        model.eval()
        transforms = [
            ChemicalSpeciesToAtomTypeMapper(
                model_type_names=["O", "H"],
                chemical_species_to_atom_type_map={"O": "O", "H": "H"},
            ),
            NeighborListTransform(r_max=3.0, type_names=["O", "H"]),
        ]
        calculator = calculator_cls(model=model, device=device, transforms=transforms)
        calculator.metadata = {
            "type_names": ["O", "H"],
            "chemical_species_to_atom_type_map": {"O": "O", "H": "H"},
        }
        print("model load status: ok (ModelFromCheckpoint + NequIPCalculator)")
        return calculator, None, False
    except Exception as exc:
        message = f"ModelFromCheckpoint: {type(exc).__name__}: {exc}"
        load_errors.append(message)
        if _looks_like_element_mismatch(message):
            return None, message, True

    return None, "\n".join(load_errors), False


def main() -> int:
    try:
        print(f"Python version: {sys.version.split()[0]}")
        print(f"Python executable: {sys.executable}")
        print(f"Conda environment: {os.environ.get('CONDA_DEFAULT_ENV', '<unset>')}")

        torch, torch_error = _import_required("torch")
        nequip, nequip_error = _import_required("nequip")
        allegro, allegro_error = _import_required("allegro")
        ase_module, ase_error = _import_required("ase")
        print(f"torch import status: {'ok' if torch is not None else f'failed ({torch_error})'}")
        print(f"nequip import status: {'ok' if nequip is not None else f'failed ({nequip_error})'}")
        print(f"allegro import status: {'ok' if allegro is not None else f'failed ({allegro_error})'}")
        print(f"ase import status: {'ok' if ase_module is not None else f'failed ({ase_error})'}")
        if torch is None or nequip is None or allegro is None or ase_module is None:
            return EXIT_IMPORT_ERROR

        print(f"torch version: {getattr(torch, '__version__', 'unknown')}")
        device, device_error = _select_device(torch)
        if device_error is not None:
            return device_error
        print(f"selected device: {device}")

        atoms = _build_h2o(ase_module)
        print(f"H2O structure: ok (symbols={atoms.get_chemical_symbols()}, pbc={atoms.pbc.tolist()})")

        required = os.environ.get("ALLEGRO_INFERENCE_REQUIRED", "0") == "1"
        model_path_env = os.environ.get("ALLEGRO_MODEL_PATH")
        if not model_path_env:
            print("No model path provided; skipping H2O model inference smoke test.")
            return EXIT_MODEL_PATH_ERROR if required else EXIT_OK

        model_path = Path(model_path_env).expanduser()
        print(f"model path: {model_path}")
        if not model_path.is_file():
            print(f"error: ALLEGRO_MODEL_PATH does not exist or is not a file: {model_path}")
            return EXIT_MODEL_PATH_ERROR

        calculator_cls, calculator_error = _calculator_class()
        if calculator_cls is None:
            print(f"model load status: failed ({calculator_error})")
            return EXIT_MODEL_LOAD_FAILED

        calculator, load_error, element_mismatch = _try_load_calculator(
            calculator_cls, model_path, device
        )
        if calculator is None:
            if element_mismatch:
                print(
                    "The provided Allegro/NequIP model does not support H/O elements "
                    "required by the H2O smoke test."
                )
                print("failure type: model element mismatch")
                print(load_error)
                return EXIT_ELEMENT_MISMATCH
            print("model load status: failed")
            print("failure type: model format/API mismatch or unsupported loader")
            print(load_error)
            return EXIT_MODEL_LOAD_FAILED

        metadata = getattr(calculator, "metadata", None)
        symbols = _metadata_symbols(metadata)
        if symbols:
            print(f"model supported symbols metadata: {sorted(symbols)}")
            if not REQUIRED_SYMBOLS.issubset(symbols):
                print(
                    "The provided Allegro/NequIP model does not support H/O elements "
                    "required by the H2O smoke test."
                )
                print("failure type: model element mismatch")
                return EXIT_ELEMENT_MISMATCH
        else:
            print("model supported symbols metadata: unavailable")

        atoms.calc = calculator
        try:
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            if _looks_like_element_mismatch(message):
                print(
                    "The provided Allegro/NequIP model does not support H/O elements "
                    "required by the H2O smoke test."
                )
                print("failure type: model element mismatch")
                print(message)
                return EXIT_ELEMENT_MISMATCH
            print(f"inference status: failed ({message})")
            if device == "mps":
                print("MPS inference failed; retry this smoke test with ALLEGRO_SMOKE_DEVICE=cpu.")
            return EXIT_INFERENCE_FAILED

        print(f"water total energy: {energy}")
        print(f"water forces:\n{forces}")
        print(f"water forces shape: {forces.shape}")
        print(f"device: {device}")
        print(f"dtype: {getattr(forces, 'dtype', '<unknown>')}")
        if tuple(forces.shape) != (3, 3):
            print("inference status: failed (force shape is not (3, 3))")
            return EXIT_INFERENCE_FAILED
        print("inference status: ok")
        return EXIT_OK
    except Exception as exc:  # pragma: no cover - defensive real-env error path
        print(f"runtime error: {type(exc).__name__}: {exc}")
        return EXIT_RUNTIME_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
