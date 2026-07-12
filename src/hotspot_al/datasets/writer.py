"""Backend-neutral serialization of labeled hotspot regions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from ase.io import write

from hotspot_al.models import ExtractedRegion
from hotspot_al.training.mask_generator import generate_region_labels
from hotspot_al.training.region_codes import region_codes_for_labels, region_label_map_json


def write_dataset_entry(
    region: ExtractedRegion,
    *,
    forces: np.ndarray,
    mask: np.ndarray,
    output_dir: str | Path,
    prefix: str,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Write one backend-neutral hotspot training sample."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    atoms = region.atoms.copy()
    force_array = np.asarray(forces, dtype=float)
    mask_array = np.asarray(mask, dtype=float)
    expected_force_shape = (len(atoms), 3)
    if force_array.shape != expected_force_shape:
        msg = f"Expected forces with shape {expected_force_shape}, got {force_array.shape}."
        raise ValueError(msg)
    expected_mask_shape = (len(atoms),)
    if mask_array.shape != expected_mask_shape:
        msg = f"Expected mask with shape {expected_mask_shape}, got {mask_array.shape}."
        raise ValueError(msg)
    atoms.arrays["forces"] = force_array
    atoms.arrays["mask_weights"] = mask_array
    region_labels = generate_region_labels(region)
    atoms.arrays["region_code"] = region_codes_for_labels(region_labels)
    atoms.info["region_label_map"] = region_label_map_json()
    if region.metadata.get("original_frame_id") is not None:
        atoms.info["original_frame_id"] = region.metadata.get("original_frame_id")
    if region.metadata.get("hotspot_id") is not None:
        atoms.info["hotspot_id"] = region.metadata.get("hotspot_id")

    xyz_path = target / f"{prefix}.extxyz"
    write(xyz_path, atoms, format="extxyz")

    npz_path = target / f"{prefix}_labels.npz"
    np.savez(
        npz_path,
        forces=force_array,
        mask=mask_array,
        original_indices=np.asarray(region.original_indices, dtype=int),
        core_indices=np.asarray(region.core_indices, dtype=int),
        inner_buffer_indices=np.asarray(region.inner_buffer_indices, dtype=int),
        outer_buffer_indices=np.asarray(region.outer_buffer_indices, dtype=int),
        boundary_indices=np.asarray(region.boundary_indices, dtype=int),
        h_cap_indices=np.asarray(region.h_cap_indices, dtype=int),
    )

    metadata_path = target / f"{prefix}_metadata.json"
    metadata = {
        "region_labels": region_labels,
        "atom_role": region_labels,
        "force_weight": mask_array.tolist(),
        "energy_weight": float((extra_metadata or {}).get("energy_weight", 0.0)),
        "core_atom_indices": list(region.core_indices),
        "masked_atom_indices": np.where(mask_array <= 0.0)[0].astype(int).tolist(),
        "metadata": region.metadata,
        "extra_metadata": extra_metadata or {},
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"structure": xyz_path, "labels": npz_path, "metadata": metadata_path}


__all__ = ["write_dataset_entry"]
