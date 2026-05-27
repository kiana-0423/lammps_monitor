#!/usr/bin/env python
"""Generate a tiny synthetic H/O dataset for Allegro/NequIP smoke tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write


DATASET_PATH = Path("tests/fixtures/allegro/tiny_h2o_synthetic.extxyz")


BASE_POSITIONS = np.array(
    [
        [0.000000, 0.000000, 0.000000],
        [0.958400, 0.000000, 0.000000],
        [-0.239000, 0.927000, 0.000000],
    ],
    dtype=float,
)


def synthetic_energy_forces(positions: np.ndarray) -> tuple[float, np.ndarray]:
    """Toy spring energy and analytic forces; not physically meaningful."""
    pairs = [
        (0, 1, 0.9584, 1.0),
        (0, 2, 0.9584, 1.0),
        (1, 2, 1.5140, 0.2),
    ]
    energy = 0.0
    forces = np.zeros_like(positions)
    for i, j, r0, k in pairs:
        rij = positions[j] - positions[i]
        dist = float(np.linalg.norm(rij))
        if dist == 0.0:
            continue
        delta = dist - r0
        unit = rij / dist
        energy += 0.5 * k * delta * delta
        force_j = -k * delta * unit
        forces[j] += force_j
        forces[i] -= force_j
    return energy, forces


def build_structures() -> list[Atoms]:
    rng = np.random.default_rng(7)
    structures: list[Atoms] = []
    for idx in range(12):
        perturbation = rng.normal(loc=0.0, scale=0.025, size=BASE_POSITIONS.shape)
        perturbation[0] *= 0.3
        positions = BASE_POSITIONS + perturbation
        atoms = Atoms("OH2", positions=positions)
        atoms.pbc = False
        energy, forces = synthetic_energy_forces(positions)
        atoms.info["total_energy"] = float(energy)
        atoms.info["toy_synthetic"] = True
        atoms.info["toy_index"] = idx
        atoms.calc = SinglePointCalculator(atoms, energy=float(energy), forces=forces)
        structures.append(atoms)
    return structures


def main() -> int:
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    structures = build_structures()
    write(DATASET_PATH, structures, format="extxyz")
    print(f"tiny H2O synthetic dataset: {DATASET_PATH}")
    print(f"num structures: {len(structures)}")
    print("elements: H, O")
    print("energy field: total_energy / calculator energy")
    print("forces field: forces")
    print("note: toy/synthetic smoke-test data only; not DFT and not physically meaningful")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
