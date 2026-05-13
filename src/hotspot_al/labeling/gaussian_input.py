"""Gaussian input generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hotspot_al.models import ExtractedRegion


def build_gaussian_input(region: ExtractedRegion, *, config: dict[str, Any], optimize_h_only: bool = False) -> str:
    """Build a minimal Gaussian input file."""

    dft_cfg = config.get("dft", config)
    route = f"#p {dft_cfg.get('functional', 'PBE')}/{dft_cfg.get('basis', 'DZVP')} force"
    if optimize_h_only and region.h_cap_indices:
        route = f"#p {dft_cfg.get('functional', 'PBE')}/{dft_cfg.get('basis', 'DZVP')} opt=modredundant"
    lines = [
        route,
        "",
        "Hotspot region",
        "",
        f"{int(dft_cfg.get('charge', 0))} {int(dft_cfg.get('multiplicity', 1))}",
    ]
    for symbol, position in zip(region.atoms.get_chemical_symbols(), region.atoms.get_positions(), strict=True):
        lines.append(f"{symbol:2s} {position[0]: .8f} {position[1]: .8f} {position[2]: .8f}")
    lines.append("")
    return "\n".join(lines)


def write_gaussian_inputs(
    region: ExtractedRegion,
    output_dir: str | Path,
    *,
    config: dict[str, Any],
    job_name: str = "hotspot_region",
) -> dict[str, Path]:
    """Write Gaussian inputs for an extracted region."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    sp_path = target / f"{job_name}.gjf"
    sp_path.write_text(build_gaussian_input(region, config=config), encoding="utf-8")
    return {"single_point_input": sp_path}
