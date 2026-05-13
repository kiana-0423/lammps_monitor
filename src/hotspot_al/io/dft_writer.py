"""Dispatchers for writing DFT input files from extracted regions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ase.io import write

from hotspot_al.cp2k.cp2k_input import write_cp2k_inputs
from hotspot_al.labeling.gaussian_input import write_gaussian_inputs
from hotspot_al.labeling.orca_input import write_orca_inputs
from hotspot_al.models import ExtractedRegion


def write_dft_inputs(
    region: ExtractedRegion,
    output_dir: str | Path,
    *,
    engine: str,
    config: dict[str, Any],
    job_name: str = "hotspot_region",
) -> dict[str, Path]:
    """Write DFT inputs for a single extracted region."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    structure_path = target / f"{job_name}.xyz"
    write(structure_path, region.atoms)

    if engine.lower() == "cp2k":
        written = write_cp2k_inputs(region, target, config=config, job_name=job_name)
    elif engine.lower() == "gaussian":
        written = write_gaussian_inputs(region, target, config=config, job_name=job_name)
    elif engine.lower() == "orca":
        written = write_orca_inputs(region, target, config=config, job_name=job_name)
    else:
        msg = f"Unsupported DFT engine: {engine}"
        raise ValueError(msg)

    written["structure"] = structure_path
    return written
