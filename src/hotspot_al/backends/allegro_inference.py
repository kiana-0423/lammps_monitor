"""Online Allegro/NequIP inference adapter.

The module keeps heavyweight Allegro dependencies optional. Importing it does
not require torch or nequip; constructing an evaluator for real inference does.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms

from hotspot_al.training.allegro_runner import ForceEvaluator


class AllegroInference:
    """Cache deployed Allegro models and expose ``AllegroRunner`` evaluators."""

    def __init__(
        self,
        model_paths: list[str | Path],
        *,
        device: str = "auto",
        type_map: dict[int, str] | None = None,
    ) -> None:
        if not model_paths:
            msg = "AllegroInference requires at least one deployed model path."
            raise ValueError(msg)
        self.model_paths = [Path(path) for path in model_paths]
        self.device = self._resolve_device(device)
        self.type_map = type_map or {}
        self._cache: dict[Path, Any] = {}

    def _resolve_device(self, device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch
        except ImportError:
            return "cpu"
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    def _load_model(self, model_path: str | Path | None) -> Any:
        path = Path(model_path or self.model_paths[0])
        if not path.is_file():
            msg = f"Allegro deployed model does not exist: {path}"
            raise FileNotFoundError(msg)
        if path in self._cache:
            return self._cache[path]

        try:
            from nequip.deploy import load_deployed_model
        except ImportError:
            load_deployed_model = None

        if load_deployed_model is not None:
            loaded = load_deployed_model(path, device=self.device)
            model = loaded[0] if isinstance(loaded, tuple) else loaded
        else:
            try:
                import torch
            except ImportError as exc:
                msg = "Online Allegro inference requires torch, or nequip with nequip.deploy available."
                raise ImportError(msg) from exc
            model = torch.jit.load(str(path), map_location=self.device)

        if hasattr(model, "eval"):
            model.eval()
        self._cache[path] = model
        return model

    def make_evaluator(self) -> ForceEvaluator:
        """Return a callback compatible with ``AllegroRunner(force_evaluator=...)``."""

        def evaluate(atoms: Atoms, model_path: str | Path | None, config: dict[str, Any]) -> np.ndarray:
            return self.predict_forces(atoms, model_path=model_path, config=config)

        return evaluate

    def reload(self, model_paths: list[str | Path] | None = None) -> None:
        """Drop cached model objects and optionally switch deployed paths."""

        if model_paths is not None:
            if not model_paths:
                msg = "AllegroInference.reload requires at least one model path when paths are provided."
                raise ValueError(msg)
            self.model_paths = [Path(path) for path in model_paths]
        self._cache.clear()

    def predict_forces(
        self,
        atoms: Atoms,
        *,
        model_path: str | Path | None = None,
        config: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Predict forces for one model, returning NaNs if runtime inference fails."""

        try:
            model = self._load_model(model_path)
            return self._call_model(model, atoms, config=config or {})
        except Exception:
            return np.full((len(atoms), 3), np.nan, dtype=float)

    def predict_committee(self, atoms: Atoms, *, config: dict[str, Any] | None = None) -> np.ndarray:
        """Predict forces for all configured models."""

        return np.stack(
            [self.predict_forces(atoms, model_path=path, config=config or {}) for path in self.model_paths],
            axis=0,
        )

    def _call_model(self, model: Any, atoms: Atoms, *, config: dict[str, Any]) -> np.ndarray:
        try:
            import torch
        except ImportError as exc:
            msg = "Online Allegro inference requires torch at prediction time."
            raise ImportError(msg) from exc

        data = self._atoms_to_model_input(atoms, torch=torch, config=config)
        with torch.no_grad():
            output = model(data)
        forces = self._extract_forces(output)
        forces_array = np.asarray(forces, dtype=float)
        expected_shape = (len(atoms), 3)
        if forces_array.shape != expected_shape:
            msg = f"Expected Allegro forces with shape {expected_shape}, got {forces_array.shape}."
            raise ValueError(msg)
        return forces_array

    def _atoms_to_model_input(self, atoms: Atoms, *, torch: Any, config: dict[str, Any]) -> dict[str, Any]:
        positions = torch.as_tensor(atoms.get_positions(), dtype=torch.get_default_dtype(), device=self.device)
        cell = torch.as_tensor(atoms.cell.array, dtype=torch.get_default_dtype(), device=self.device).unsqueeze(0)
        atomic_numbers = torch.as_tensor(atoms.get_atomic_numbers(), dtype=torch.long, device=self.device)
        pbc = torch.as_tensor(np.asarray(atoms.pbc, dtype=bool), dtype=torch.bool, device=self.device).unsqueeze(0)
        batch = torch.zeros(len(atoms), dtype=torch.long, device=self.device)
        return {
            "pos": positions,
            "positions": positions,
            "cell": cell,
            "atomic_numbers": atomic_numbers,
            "atom_types": atomic_numbers,
            "pbc": pbc,
            "batch": batch,
            **dict(config.get("allegro_model_input", {})),
        }

    def _extract_forces(self, output: Any) -> Any:
        if isinstance(output, dict):
            for key in ("forces", "force", "atomic_forces"):
                if key in output:
                    value = output[key]
                    if hasattr(value, "detach"):
                        return value.detach().cpu().numpy()
                    return value
        if hasattr(output, "forces"):
            value = output.forces
            if hasattr(value, "detach"):
                return value.detach().cpu().numpy()
            return value
        msg = "Allegro model output did not contain a forces field."
        raise ValueError(msg)
