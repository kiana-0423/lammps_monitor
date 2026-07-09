"""Optional real LAMMPS integration test with a tiny LJ system."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from hotspot_al.io.lammps_reader import read_dump

pytestmark = [pytest.mark.integration, pytest.mark.lammps]


def _lammps_bin_or_skip() -> str:
    if os.environ.get("RUN_EXTERNAL") != "1":
        pytest.skip("Set RUN_EXTERNAL=1 to enable real LAMMPS integration tests.")
    lammps_bin = os.environ.get("LAMMPS_BIN")
    if not lammps_bin:
        pytest.skip("LAMMPS_BIN is not set.")
    return lammps_bin


def test_lammps_lj_dump_is_readable(tmp_path: Path) -> None:
    lammps_bin = _lammps_bin_or_skip()
    input_path = tmp_path / "lj.in"
    dump_path = tmp_path / "lj.dump"
    input_path.write_text(
        f"""units lj
atom_style atomic
boundary p p p
region box block 0 5 0 5 0 5
create_box 1 box
create_atoms 1 single 1.0 1.0 1.0
create_atoms 1 single 1.5 1.0 1.0
mass 1 1.0
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0 2.5
neighbor 0.3 bin
neigh_modify every 1 delay 0 check yes
dump d all custom 1 {dump_path.name} id type x y z fx fy fz
run 0
""",
        encoding="utf-8",
    )

    result = subprocess.run(
        [lammps_bin, "-in", input_path.name],
        check=False,
        text=True,
        capture_output=True,
        cwd=tmp_path,
        timeout=60,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    frames = read_dump(dump_path, type_map={1: "Ar"})
    assert frames
    frame = frames[0]
    assert len(frame.atoms) == 2
    assert frame.atoms.get_positions().shape == (2, 3)
    assert frame.forces is not None
    assert frame.forces.shape == (2, 3)
