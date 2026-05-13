"""Generate atom-wise force supervision masks for extracted regions."""

from __future__ import annotations

import numpy as np

from hotspot_al.extraction.boundary_regions import region_label_vector
from hotspot_al.models import ExtractedRegion


def generate_atom_mask(region: ExtractedRegion, config: dict) -> np.ndarray:
    """Create an atom-wise training weight mask from region labels."""

    mask_cfg = config.get("training_mask", config)
    mask = np.zeros(len(region.atoms), dtype=float)
    mask[region.outer_buffer_indices] = float(mask_cfg.get("outer_buffer", 0.0))
    mask[region.boundary_indices] = float(mask_cfg.get("boundary", 0.0))
    mask[region.inner_buffer_indices] = float(mask_cfg.get("inner_buffer", 0.3))
    mask[region.core_indices] = float(mask_cfg.get("core", 1.0))
    mask[region.h_cap_indices] = float(mask_cfg.get("h_cap", 0.0))
    region.mask_weights = mask.copy()
    if not region.region_labels:
        region.region_labels = generate_region_labels(region)
    return mask


def generate_region_labels(region: ExtractedRegion) -> list[str]:
    """Return a human-readable per-atom region label list."""

    regions = {
        "core": region.core_indices,
        "inner_buffer": region.inner_buffer_indices,
        "outer_buffer": region.outer_buffer_indices,
        "boundary": region.boundary_indices,
    }
    labels = region_label_vector(len(region.atoms), regions)
    for index in region.h_cap_indices:
        labels[index] = "h_cap"
    return labels
