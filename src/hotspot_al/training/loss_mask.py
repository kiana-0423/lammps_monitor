"""Reference masked force loss utilities."""

from __future__ import annotations

import numpy as np


def masked_force_mse(predicted: np.ndarray, target: np.ndarray, mask: np.ndarray, epsilon: float = 1.0e-8) -> float:
    """Compute a mask-weighted mean squared force loss."""

    predicted = np.asarray(predicted, dtype=float)
    target = np.asarray(target, dtype=float)
    mask = np.asarray(mask, dtype=float).reshape(-1, 1)
    squared = np.sum((predicted - target) ** 2, axis=1, keepdims=True)
    weighted = mask * squared
    return float(np.sum(weighted) / (np.sum(mask) + epsilon))
