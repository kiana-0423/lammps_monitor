"""ORCA input generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hotspot_al.models import ExtractedRegion


def build_orca_input(region: ExtractedRegion, *, config: dict[str, Any]) -> str:
    """Build a minimal ORCA force job input."""

    dft_cfg = config.get("dft", config)
    header = f"! {dft_cfg.get('functional', 'PBE')} {dft_cfg.get('basis', 'def2-SVP')} EnGrad"
    lines = [
        header,
        "",
        f"* xyz {int(dft_cfg.get('charge', 0))} {int(dft_cfg.get('multiplicity', 1))}",
    ]
    for symbol, position in zip(region.atoms.get_chemical_symbols(), region.atoms.get_positions(), strict=True):
        lines.append(f"  {symbol:2s} {position[0]: .8f} {position[1]: .8f} {position[2]: .8f}")
    lines.append("*")
    return "\n".join(lines)


def write_orca_inputs(
    region: ExtractedRegion,
    output_dir: str | Path,
    *,
    config: dict[str, Any],
    job_name: str = "hotspot_region",
) -> dict[str, Path]:
    """Write ORCA inputs for an extracted region."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    sp_path = target / f"{job_name}.inp"
    sp_path.write_text(build_orca_input(region, config=config), encoding="utf-8")
    return {"single_point_input": sp_path}
