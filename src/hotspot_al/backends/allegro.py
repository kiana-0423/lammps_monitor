"""Real Allegro backend skeleton."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms

from hotspot_al.training.allegro_runner import AllegroRunner


class RealAllegroBackend:
    """Compatibility wrapper around ``AllegroRunner`` for future real runtimes."""

    def __init__(self, *, runner: AllegroRunner | None = None) -> None:
        self.runner = runner or AllegroRunner()

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "RealAllegroBackend":
        """Build a backend from config using the real Allegro inference adapter."""

        return cls(runner=AllegroRunner.from_config(config))

    def evaluate_forces(
        self,
        atoms: Atoms,
        *,
        config: dict[str, Any] | None = None,
        model_path: str | Path | None = None,
    ) -> np.ndarray:
        """Delegate to the existing AllegroRunner API."""

        return self.runner.evaluate_forces(atoms, config=config or {}, model_path=model_path)

    def evaluate_committee(
        self,
        atoms: Atoms,
        *,
        config: dict[str, Any] | None = None,
        model_paths: list[str],
    ) -> np.ndarray:
        """Delegate committee evaluation to the existing AllegroRunner API."""

        return self.runner.evaluate_committee(atoms, config=config or {}, model_paths=model_paths)
