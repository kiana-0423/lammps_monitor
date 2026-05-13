"""Abstract backend interface for Allegro active learning hooks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ase import Atoms


class MLIPBackend(ABC):
    """Minimal backend interface used by the Allegro wrapper."""

    name: str

    @abstractmethod
    def write_lammps_input(self, config: dict[str, Any]) -> str:
        """Build the backend-specific LAMMPS pair_style block."""

    def evaluate_forces(self, atoms: Atoms):
        """Evaluate forces for one structure.

        The baseline project leaves actual model inference to the external
        runtime and exposes this method as an extension hook.
        """

        msg = f"{self.name} force evaluation is not implemented in the baseline package."
        raise NotImplementedError(msg)

    def evaluate_committee(self, atoms: Atoms, model_paths: list[str]):
        """Evaluate a committee of models on one structure."""

        msg = f"{self.name} committee evaluation is not implemented in the baseline package."
        raise NotImplementedError(msg)

    @abstractmethod
    def write_training_data(self, *args, **kwargs):
        """Write backend-specific training data."""

    def train(self, config: dict[str, Any]):
        """Launch training. This remains an integration hook."""

        msg = f"{self.name} training launch is not implemented in the baseline package."
        raise NotImplementedError(msg)

    def export_model(self, output_dir: str | Path):
        """Export a deployable model artifact."""

        msg = f"{self.name} model export is not implemented in the baseline package."
        raise NotImplementedError(msg)
