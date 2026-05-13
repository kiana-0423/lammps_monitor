"""CP2K input generation for hotspot-localized DFT labeling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hotspot_al.cp2k.cp2k_templates import constraint_block, global_block
from hotspot_al.models import ExtractedRegion


def _cp2k_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("cp2k", config.get("dft", config))


def build_cp2k_input(
    region: ExtractedRegion,
    *,
    config: dict[str, Any],
    run_type: str,
    fixed_atom_indices: list[int] | None = None,
    project_name: str = "hotspot_region",
) -> str:
    """Build a CP2K input file for H-only optimization or force single-point."""

    cp2k_cfg = _cp2k_cfg(config)
    unique_symbols = sorted(set(region.atoms.get_chemical_symbols()))
    periodic = str(cp2k_cfg.get("periodic", "NONE"))
    cell = region.atoms.cell.array

    kind_blocks = []
    for symbol in unique_symbols:
        kind_blocks.append(
            "\n".join(
                [
                    f"    &KIND {symbol}",
                    f"      BASIS_SET {cp2k_cfg.get('basis', 'DZVP-MOLOPT-SR-GTH')}",
                    f"      POTENTIAL {cp2k_cfg.get('potential', 'GTH-PBE')}",
                    "    &END KIND",
                ]
            )
        )

    motion_block = ""
    if run_type.upper() == "GEO_OPT":
        motion_block = "\n".join(
            [
                "&MOTION",
                "  &GEO_OPT",
                "    OPTIMIZER BFGS",
                f"    MAX_ITER {int(cp2k_cfg.get('h_only_opt', {}).get('max_iter', 50))}",
                "  &END GEO_OPT",
                constraint_block(fixed_atom_indices),
                "&END MOTION",
            ]
        )

    coord_lines = [
        f"    {symbol:<2} {position[0]:>16.8f} {position[1]:>16.8f} {position[2]:>16.8f}"
        for symbol, position in zip(region.atoms.get_chemical_symbols(), region.atoms.get_positions(), strict=True)
    ]

    ot_block = ""
    if bool(cp2k_cfg.get("use_ot", True)):
        ot_block = "\n".join(
            [
                "      &OT",
                "        MINIMIZER DIIS",
                "        PRECONDITIONER FULL_SINGLE_INVERSE",
                "      &END OT",
            ]
        )

    dispersion_block = ""
    if bool(cp2k_cfg.get("dispersion", False)):
        dispersion_block = "\n".join(
            [
                "      &VDW_POTENTIAL",
                "        POTENTIAL_TYPE PAIR_POTENTIAL",
                "        &PAIR_POTENTIAL",
                "          TYPE DFTD3(BJ)",
                "          PARAMETER_FILE_NAME dftd3.dat",
                f"          REFERENCE_FUNCTIONAL {cp2k_cfg.get('functional', 'PBE')}",
                "        &END PAIR_POTENTIAL",
                "      &END VDW_POTENTIAL",
            ]
        )

    return "\n".join(
        [
            global_block(run_type, project_name=project_name),
            "&FORCE_EVAL",
            "  METHOD Quickstep",
            "  &DFT",
            f"    CHARGE {int(cp2k_cfg.get('charge', 0))}",
            f"    MULTIPLICITY {int(cp2k_cfg.get('multiplicity', 1))}",
            "    BASIS_SET_FILE_NAME BASIS_MOLOPT",
            "    POTENTIAL_FILE_NAME GTH_POTENTIALS",
            "    &MGRID",
            f"      CUTOFF {float(cp2k_cfg.get('cutoff', 400))}",
            f"      REL_CUTOFF {float(cp2k_cfg.get('rel_cutoff', 60))}",
            "    &END MGRID",
            "    &QS",
            "      EPS_DEFAULT 1.0E-12",
            "    &END QS",
            "    &POISSON",
            f"      PERIODIC {periodic}",
            f"      POISSON_SOLVER {cp2k_cfg.get('poisson_solver', 'WAVELET')}",
            "    &END POISSON",
            "    &SCF",
            f"      MAX_SCF {int(cp2k_cfg.get('max_scf', 100))}",
            f"      EPS_SCF {cp2k_cfg.get('scf_eps', 1.0e-6)}",
            "      SCF_GUESS ATOMIC",
            ot_block,
            "    &END SCF",
            "    &XC",
            f"      &XC_FUNCTIONAL {cp2k_cfg.get('functional', 'PBE')}",
            "      &END XC_FUNCTIONAL",
            dispersion_block,
            "    &END XC",
            "    &PRINT",
            "      &FORCES ON",
            "      &END FORCES",
            "    &END PRINT",
            "  &END DFT",
            "  &SUBSYS",
            "    &CELL",
            f"      A {cell[0, 0]:.8f} {cell[0, 1]:.8f} {cell[0, 2]:.8f}",
            f"      B {cell[1, 0]:.8f} {cell[1, 1]:.8f} {cell[1, 2]:.8f}",
            f"      C {cell[2, 0]:.8f} {cell[2, 1]:.8f} {cell[2, 2]:.8f}",
            f"      PERIODIC {periodic}",
            "    &END CELL",
            "    &COORD",
            *coord_lines,
            "    &END COORD",
            *kind_blocks,
            "  &END SUBSYS",
            "&END FORCE_EVAL",
            motion_block,
            "",
        ]
    )


def write_cp2k_inputs(
    region: ExtractedRegion,
    output_dir: str | Path,
    *,
    config: dict[str, Any],
    job_name: str = "hotspot_region",
) -> dict[str, Path]:
    """Write CP2K inputs for H-only optimization and single-point forces."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    cp2k_cfg = _cp2k_cfg(config)
    h_opt_cfg = cp2k_cfg.get("h_only_opt", {})
    sp_cfg = cp2k_cfg.get("single_point", {})

    if bool(h_opt_cfg.get("enabled", True)) and region.h_cap_indices:
        fixed_atoms = [index for index in range(len(region.atoms)) if index not in set(region.h_cap_indices)]
        hopt_path = target / f"{job_name}_hopt.inp"
        hopt_path.write_text(
            build_cp2k_input(
                region,
                config=config,
                run_type="GEO_OPT",
                fixed_atom_indices=fixed_atoms,
                project_name=f"{job_name}_hopt",
            ),
            encoding="utf-8",
        )
        written["hopt_input"] = hopt_path

    if bool(sp_cfg.get("enabled", True)):
        sp_path = target / f"{job_name}_sp.inp"
        sp_path.write_text(
            build_cp2k_input(
                region,
                config=config,
                run_type="ENERGY_FORCE",
                fixed_atom_indices=None,
                project_name=f"{job_name}_sp",
            ),
            encoding="utf-8",
        )
        written["single_point_input"] = sp_path
    return written
