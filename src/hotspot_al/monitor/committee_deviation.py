"""Committee-model disagreement metrics."""

from __future__ import annotations

import numpy as np


def committee_force_deviation(force_predictions: np.ndarray) -> np.ndarray:
    """Return per-atom committee force deviation.

    Parameters
    ----------
    force_predictions
        Array with shape ``(n_models, n_atoms, 3)``.
    """

    predictions = np.asarray(force_predictions, dtype=float)
    if predictions.ndim != 3:
        msg = f"Expected committee predictions with shape (n_models, n_atoms, 3), got {predictions.shape}"
        raise ValueError(msg)
    mean_forces = np.mean(predictions, axis=0)
    squared = np.sum((predictions - mean_forces) ** 2, axis=-1)
    return np.sqrt(np.mean(squared, axis=0))
