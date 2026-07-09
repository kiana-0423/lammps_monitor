"""Allegro dataset export helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from ase.io import write

from hotspot_al.models import ExtractedRegion
from hotspot_al.training.mask_generator import generate_atom_mask, generate_region_labels
from hotspot_al.training.region_codes import region_codes_for_labels, region_label_map_json


def write_allegro_dataset(
    region: ExtractedRegion,
    *,
    forces: np.ndarray,
    output_dir: str | Path,
    config: dict[str, Any],
    filename: str = "dataset.extxyz",
) -> dict[str, Path]:
    """Write one extracted region into an Allegro-friendly extxyz."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    atoms = region.atoms.copy()
    mask = generate_atom_mask(region, config)
    atoms.arrays["forces"] = np.asarray(forces, dtype=float)
    atoms.arrays["mask_weights"] = mask
    atoms.arrays["region_code"] = region_codes_for_labels(generate_region_labels(region))
    atoms.info["mask_weight_key"] = "mask_weights"
    atoms.info["region_label_map"] = region_label_map_json()
    if region.metadata.get("event_id") is not None:
        atoms.info["event_id"] = region.metadata.get("event_id")
    path = target / filename
    write(path, atoms, format="extxyz")
    metadata_path = target / f"{path.stem}_metadata.json"
    metadata_path.write_text(json.dumps(region.metadata, indent=2), encoding="utf-8")
    return {"dataset": path, "metadata": metadata_path}
