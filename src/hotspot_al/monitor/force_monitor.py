"""Force-based atom-wise monitors."""

from __future__ import annotations

import numpy as np

from hotspot_al.utils.geometry import row_norms


def force_norms(forces: np.ndarray) -> np.ndarray:
    """Return per-atom force magnitudes."""

    return row_norms(np.asarray(forces, dtype=float))


def delta_force_norms(current_forces: np.ndarray, previous_forces: np.ndarray | None) -> np.ndarray:
    """Return per-atom force jumps between consecutive frames."""

    current = np.asarray(current_forces, dtype=float)
    if previous_forces is None:
        return np.zeros(len(current), dtype=float)
    previous = np.asarray(previous_forces, dtype=float)
    return row_norms(current - previous)
