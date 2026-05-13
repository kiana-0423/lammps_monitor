"""Dataset serialization utilities for masked hotspot training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from ase.io import write

from hotspot_al.models import ExtractedRegion
from hotspot_al.training.mask_generator import generate_region_labels


def write_dataset_entry(
    region: ExtractedRegion,
    *,
    forces: np.ndarray,
    mask: np.ndarray,
    output_dir: str | Path,
    prefix: str,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Write one hotspot region as a training sample."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    atoms = region.atoms.copy()
    atoms.arrays["forces"] = np.asarray(forces, dtype=float)
    atoms.arrays["mask_weights"] = np.asarray(mask, dtype=float)
    region_codes = np.asarray(
        [
            {"core": 0, "inner_buffer": 1, "outer_buffer": 2, "boundary": 3, "h_cap": 4}.get(label, -1)
            for label in generate_region_labels(region)
        ],
        dtype=int,
    )
    atoms.arrays["region_code"] = region_codes
    atoms.info["region_label_map"] = json.dumps({0: "core", 1: "inner_buffer", 2: "outer_buffer", 3: "boundary", 4: "h_cap"})
    if region.metadata.get("original_frame_id") is not None:
        atoms.info["original_frame_id"] = region.metadata.get("original_frame_id")
    if region.metadata.get("hotspot_id") is not None:
        atoms.info["hotspot_id"] = region.metadata.get("hotspot_id")
    xyz_path = target / f"{prefix}.extxyz"
    write(xyz_path, atoms, format="extxyz")

    npz_path = target / f"{prefix}_labels.npz"
    np.savez(
        npz_path,
        forces=np.asarray(forces, dtype=float),
        mask=np.asarray(mask, dtype=float),
        original_indices=np.asarray(region.original_indices, dtype=int),
        core_indices=np.asarray(region.core_indices, dtype=int),
        inner_buffer_indices=np.asarray(region.inner_buffer_indices, dtype=int),
        outer_buffer_indices=np.asarray(region.outer_buffer_indices, dtype=int),
        boundary_indices=np.asarray(region.boundary_indices, dtype=int),
        h_cap_indices=np.asarray(region.h_cap_indices, dtype=int),
    )

    metadata_path = target / f"{prefix}_metadata.json"
    metadata = {
        "region_labels": generate_region_labels(region),
        "metadata": region.metadata,
        "extra_metadata": extra_metadata or {},
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"structure": xyz_path, "labels": npz_path, "metadata": metadata_path}
