"""Fast local Lennard-Jones residuals using neighbor lists and linear solves."""

from __future__ import annotations

import numpy as np
from ase import Atoms

from hotspot_al.monitor.lj_residual import LJFitResult, compute_lj_residuals
from hotspot_al.monitor.neighbor_utils import MonitorNeighbors


def _invalid_fit() -> LJFitResult:
    return LJFitResult(epsilon=0.0, sigma=0.0, residual=1.0, valid=False)


def fit_local_lj_force_linear(
    displacement_vectors: np.ndarray,
    target_force: np.ndarray,
    *,
    delta: float = 1.0e-8,
) -> LJFitResult:
    """Fit a local LJ force projection with a 2x2 linear least-squares solve."""

    vectors = np.asarray(displacement_vectors, dtype=float)
    force = np.asarray(target_force, dtype=float)
    if len(vectors) == 0:
        return _invalid_fit()

    distances = np.linalg.norm(vectors, axis=1)
    mask = distances > 1.0e-8
    if not np.any(mask):
        return _invalid_fit()

    vectors = vectors[mask]
    distances = distances[mask]
    a_vector = np.sum(2.0 * vectors / distances[:, None] ** 14, axis=0)
    b_vector = np.sum(-vectors / distances[:, None] ** 8, axis=0)
    gram = np.array(
        [
            [float(np.dot(a_vector, a_vector)), float(np.dot(a_vector, b_vector))],
            [float(np.dot(b_vector, a_vector)), float(np.dot(b_vector, b_vector))],
        ],
        dtype=float,
    )
    rhs = np.array([float(np.dot(a_vector, force)), float(np.dot(b_vector, force))], dtype=float)
    det = float(np.linalg.det(gram))
    if abs(det) < 1.0e-12:
        return _invalid_fit()

    try:
        alpha, beta = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        return _invalid_fit()
    if alpha <= 0.0 or beta <= 0.0:
        return _invalid_fit()

    sigma = float((alpha / beta) ** (1.0 / 6.0))
    if not np.isfinite(sigma) or sigma < 0.2 or sigma > 10.0:
        return _invalid_fit()

    epsilon = float(beta / (24.0 * sigma**6))
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        return _invalid_fit()

    predicted = alpha * a_vector + beta * b_vector
    residual = float(np.linalg.norm(predicted - force) / (np.linalg.norm(force) + delta))
    return LJFitResult(epsilon=epsilon, sigma=sigma, residual=residual, valid=True)


def compute_lj_residuals_fast(
    atoms: Atoms,
    forces: np.ndarray,
    nl: MonitorNeighbors | None = None,
    *,
    cutoff: float = 6.0,
    delta: float = 1.0e-8,
    suspicious_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, list[LJFitResult]]:
    """Compute LJ residuals using a neighbor list and optional lazy atom mask."""

    if nl is None:
        residuals, fits = compute_lj_residuals(atoms, forces, cutoff=cutoff, delta=delta)
        if suspicious_mask is None:
            return residuals, fits
        mask = np.asarray(suspicious_mask, dtype=bool)
        residuals = np.where(mask, residuals, 0.0)
        fits = [fit if bool(mask[i]) else LJFitResult(0.0, 0.0, 0.0, True) for i, fit in enumerate(fits)]
        return residuals, fits

    forces = np.asarray(forces, dtype=float)
    if suspicious_mask is None:
        mask = np.ones(len(atoms), dtype=bool)
    else:
        mask = np.asarray(suspicious_mask, dtype=bool)
        if len(mask) != len(atoms):
            raise ValueError("suspicious_mask length must match number of atoms.")

    residuals = np.zeros(len(atoms), dtype=float)
    fit_results: list[LJFitResult] = []
    skipped = LJFitResult(epsilon=0.0, sigma=0.0, residual=0.0, valid=True)
    for i in range(len(atoms)):
        if not bool(mask[i]):
            fit_results.append(skipped)
            continue
        _indices, displacements, _distances = nl.get_displacements(atoms, i, cutoff)
        fit = fit_local_lj_force_linear(displacements, forces[i], delta=delta)
        residuals[i] = fit.residual
        fit_results.append(fit)
    return residuals, fit_results
