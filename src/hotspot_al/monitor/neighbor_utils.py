"""Shared neighbor-list helpers for atom-wise monitors."""

from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.neighborlist import NeighborList


class MonitorNeighbors:
    """ASE-backed neighbor list shared by lightweight and physics monitors."""

    def __init__(
        self,
        atoms: Atoms,
        *,
        lj_cutoff: float = 6.0,
        coordination_cutoff: float | None = None,
        skin: float = 0.0,
    ) -> None:
        self.lj_cutoff = float(lj_cutoff)
        self.coordination_cutoff = float(coordination_cutoff if coordination_cutoff is not None else lj_cutoff)
        self.cutoff = max(self.lj_cutoff, self.coordination_cutoff)
        self.skin = float(skin)
        self.n_atoms = len(atoms)
        self._nl = self._build(atoms)

    def _build(self, atoms: Atoms) -> NeighborList:
        pair_cutoffs = np.full(len(atoms), self.cutoff * 0.5, dtype=float)
        neighbor_list = NeighborList(pair_cutoffs, skin=self.skin, self_interaction=False, bothways=True)
        neighbor_list.update(atoms)
        return neighbor_list

    def rebuild(self, atoms: Atoms) -> None:
        """Update the neighbor list for a new frame."""

        if len(atoms) != self.n_atoms:
            self.n_atoms = len(atoms)
            self._nl = self._build(atoms)
            return
        self._nl.update(atoms)

    def get_displacements(
        self,
        atoms: Atoms,
        index: int,
        sub_cutoff: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return neighbor indices, MIC displacement vectors, and distances."""

        cutoff = self.cutoff if sub_cutoff is None else float(sub_cutoff)
        indices, offsets = self._nl.get_neighbors(index)
        if len(indices) == 0:
            empty_vectors = np.empty((0, 3), dtype=float)
            return indices.astype(int), empty_vectors, np.empty(0, dtype=float)

        positions = atoms.get_positions()
        cell = atoms.cell.array
        displacements = positions[indices] + np.asarray(offsets, dtype=float) @ cell - positions[index]
        distances = np.linalg.norm(displacements, axis=1)
        mask = (distances > 1.0e-8) & (distances <= cutoff)
        return indices[mask].astype(int), displacements[mask], distances[mask]
