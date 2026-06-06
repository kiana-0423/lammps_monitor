"""Tests for LAMMPS streaming dump control helpers."""

from __future__ import annotations

from hotspot_al.lammps.lammps_controller import LAMMPSController


FRAME0 = """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
1
ITEM: BOX BOUNDS pp pp pp
0 5
0 5
0 5
ITEM: ATOMS id type x y z fx fy fz
1 1 0 0 0 1 0 0
"""

FRAME1 = """ITEM: TIMESTEP
1
ITEM: NUMBER OF ATOMS
1
ITEM: BOX BOUNDS pp pp pp
0 5
0 5
0 5
ITEM: ATOMS id type x y z fx fy fz
1 1 0.1 0 0 0.5 0 0
"""


def test_lammps_controller_waits_for_complete_streaming_frame(tmp_path) -> None:
    dump_file = tmp_path / "dump.online.lammpstrj"
    controller = LAMMPSController(
        tmp_path / "in.lammps",
        dump_file=dump_file,
        work_dir=tmp_path,
        config={"lammps": {"type_map": {1: "H"}, "timestep_fs": 0.5}},
        poll_interval=0.0,
    )

    dump_file.write_text(FRAME0, encoding="utf-8")
    assert controller.next_frame(timeout=0.0) is None

    dump_file.write_text(FRAME0 + FRAME1, encoding="utf-8")
    frame = controller.next_frame(timeout=0.0)

    assert frame is not None
    assert frame.step == 0
    assert len(frame.atoms) == 1
    assert frame.forces[0, 0] == 1.0
