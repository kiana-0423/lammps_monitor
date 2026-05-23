"""Tests for backend skeleton imports and basic error handling."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from hotspot_al.backends import CP2KBackend, LAMMPSBackend, RealAllegroBackend
from hotspot_al.training.allegro_runner import AllegroRunner


def test_backend_skeletons_import() -> None:
    assert CP2KBackend is not None
    assert LAMMPSBackend is not None
    assert RealAllegroBackend is not None


def test_cp2k_backend_evaluate_forces_requires_runner() -> None:
    atoms = Atoms(symbols=["H"], positions=[[0.0, 0.0, 0.0]])
    with pytest.raises(NotImplementedError, match="external CP2K job runner"):
        CP2KBackend().evaluate_forces(atoms)


def test_lammps_backend_evaluate_forces_requires_runtime_adapter() -> None:
    atoms = Atoms(symbols=["Ar"], positions=[[0.0, 0.0, 0.0]])
    with pytest.raises(NotImplementedError, match="external LAMMPS input/runtime adapter"):
        LAMMPSBackend().evaluate_forces(atoms)


def test_real_allegro_backend_preserves_runner_contract() -> None:
    atoms = Atoms(symbols=["H", "H"], positions=np.zeros((2, 3)))
    backend = RealAllegroBackend(runner=AllegroRunner(force_evaluator=lambda atoms, model_path, config: np.ones((len(atoms), 3))))

    forces = backend.evaluate_forces(atoms, config={}, model_path="mock.pth")

    assert forces.shape == (2, 3)
