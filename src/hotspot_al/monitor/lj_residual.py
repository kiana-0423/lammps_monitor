"""Local Lennard-Jones projection residuals for physics-aware OOD screening."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase import Atoms
from scipy.optimize import least_squares

from hotspot_al.utils.periodic import mic_displacements_from_reference


@dataclass(slots=True)
class LJFitResult:
    """Fit result for one atom."""

    epsilon: float
    sigma: float
    residual: float
    valid: bool


def _lj_pair_force(displacement: np.ndarray, epsilon: float, sigma: float) -> np.ndarray:
    radius = float(np.linalg.norm(displacement))
    if radius < 1.0e-8:
        return np.zeros(3, dtype=float)
    prefactor = 24.0 * epsilon * ((2.0 * sigma**12 / radius**14) - (sigma**6 / radius**8))
    return prefactor * displacement


def fit_local_lj_force(
    displacement_vectors: np.ndarray,
    target_force: np.ndarray,
    *,
    delta: float = 1.0e-8,
) -> LJFitResult:
    """Fit a two-parameter LJ model to a single atom's local force."""

    vectors = np.asarray(displacement_vectors, dtype=float)
    force = np.asarray(target_force, dtype=float)
    if len(vectors) == 0:
        return LJFitResult(epsilon=0.0, sigma=0.0, residual=1.0, valid=False)

    distances = np.linalg.norm(vectors, axis=1)
    positive_distances = distances[distances > 1.0e-8]
    if len(positive_distances) == 0:
        return LJFitResult(epsilon=0.0, sigma=0.0, residual=1.0, valid=False)

    sigma0 = max(float(np.median(positive_distances) / (2.0 ** (1.0 / 6.0))), 0.5)
    epsilon0 = max(float(np.linalg.norm(force) / (len(vectors) + 1.0)), 1.0e-3)
    min_distance = float(np.min(positive_distances))
    upper_sigma = max(min_distance * 1.25, 0.6)

    def residuals(log_params: np.ndarray) -> np.ndarray:
        epsilon = float(np.exp(log_params[0]))
        sigma = float(np.exp(log_params[1]))
        predicted = np.sum([_lj_pair_force(vec, epsilon, sigma) for vec in vectors], axis=0)
        return predicted - force

    result = least_squares(
        residuals,
        x0=np.log([epsilon0, min(sigma0, upper_sigma)]),
        bounds=(np.log([1.0e-8, 0.2]), np.log([1.0e4, upper_sigma])),
        max_nfev=100,
    )
    epsilon, sigma = np.exp(result.x)
    residual = float(np.linalg.norm(result.fun) / (np.linalg.norm(force) + delta))
    valid = bool(result.success and epsilon > 0.0 and 0.2 <= sigma <= upper_sigma and min_distance > 0.5)
    if not valid:
        residual = max(residual, 1.0)
    return LJFitResult(epsilon=float(epsilon), sigma=float(sigma), residual=residual, valid=valid)


def compute_lj_residuals(
    atoms: Atoms,
    forces: np.ndarray,
    *,
    cutoff: float = 6.0,
    delta: float = 1.0e-8,
) -> tuple[np.ndarray, list[LJFitResult]]:
    """Compute atom-wise LJ projection residuals."""

    positions = atoms.get_positions()
    cell = atoms.cell.array
    pbc = atoms.pbc
    forces = np.asarray(forces, dtype=float)
    residuals = np.zeros(len(atoms), dtype=float)
    fit_results: list[LJFitResult] = []
    for i, position in enumerate(positions):
        displacements = mic_displacements_from_reference(position, positions, cell=cell, pbc=pbc)
        distances = np.linalg.norm(displacements, axis=1)
        mask = (distances > 1.0e-8) & (distances <= cutoff)
        fit = fit_local_lj_force(displacements[mask], forces[i], delta=delta)
        residuals[i] = fit.residual
        fit_results.append(fit)
    return residuals, fit_results
