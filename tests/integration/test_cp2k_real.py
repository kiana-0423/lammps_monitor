"""Optional real CP2K integration test with a tiny H2 system."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from ase.io import read
import pytest

from hotspot_al.config import load_config
from hotspot_al.cp2k.cp2k_force_parser import parse_cp2k_forces
from hotspot_al.cp2k.cp2k_input import write_cp2k_inputs
from hotspot_al.models import ExtractedRegion


pytestmark = [pytest.mark.integration, pytest.mark.cp2k]


def _cp2k_bin_or_skip() -> str:
    if os.environ.get("RUN_EXTERNAL") != "1":
        pytest.skip("Set RUN_EXTERNAL=1 to enable real CP2K integration tests.")
    cp2k_bin = os.environ.get("CP2K_BIN")
    if not cp2k_bin:
        pytest.skip("CP2K_BIN is not set.")
    return cp2k_bin


def test_cp2k_energy_force_output_is_parseable(fixtures_dir: Path, tmp_path: Path) -> None:
    cp2k_bin = _cp2k_bin_or_skip()
    atoms = read(fixtures_dir / "structures" / "h2.xyz")
    atoms.cell = [8.0, 8.0, 8.0]
    atoms.pbc = False
    region = ExtractedRegion(
        atoms=atoms,
        original_indices=list(range(len(atoms))),
        core_indices=list(range(len(atoms))),
        inner_buffer_indices=[],
        outer_buffer_indices=[],
        boundary_indices=[],
        h_cap_indices=[],
        hotspot_indices=[0],
        region_labels=["core"] * len(atoms),
        metadata={"integration": "cp2k"},
    )
    config = load_config()
    config["cp2k"] = {
        **config["cp2k"],
        "executable": cp2k_bin,
        "cutoff": 120,
        "rel_cutoff": 40,
        "max_scf": 30,
        "scf_eps": 1.0e-5,
        "use_ot": True,
        "h_only_opt": {"enabled": False},
        "single_point": {"enabled": True, "print_forces": True},
    }
    written = write_cp2k_inputs(region, tmp_path, config=config, job_name="h2_real")
    input_path = written["single_point_input"]
    output_path = tmp_path / "h2_real.out"

    result = subprocess.run(
        [cp2k_bin, "-i", input_path.name, "-o", output_path.name],
        check=False,
        text=True,
        capture_output=True,
        cwd=tmp_path,
        timeout=600,
    )

    cp2k_log = output_path.read_text(encoding="utf-8", errors="replace") if output_path.exists() else ""
    assert result.returncode == 0, result.stdout + result.stderr + cp2k_log
    forces = parse_cp2k_forces(output_path)
    assert forces.shape == (len(atoms), 3)
