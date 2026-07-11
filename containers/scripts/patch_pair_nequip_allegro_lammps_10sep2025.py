"""Apply a narrow LAMMPS Kokkos typedef compatibility patch.

pair_nequip_allegro v0.7.0 uses pre-10Sep2025 Kokkos aliases such as
X_FLOAT/F_FLOAT/E_FLOAT and t_f_array. LAMMPS patch_10Sep2025 keeps the same precision
contract through KK_FLOAT/KK_ACC_FLOAT and t_kk* views. This patch is limited
to those typedef names before the third-party source is copied into LAMMPS.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPLACEMENTS = {
    "E_FLOAT": "KK_ACC_FLOAT",
    "F_FLOAT": "KK_FLOAT",
    "X_FLOAT": "KK_FLOAT",
    "t_efloat_1d": "t_kkacc_1d",
    "t_virial_array": "t_kkacc_1d_6",
    "t_x_array_randomread": "t_kkfloat_1d_3_lr_randomread",
    "t_f_array": "t_kkacc_1d_3",
    "tdual_efloat_1d": "tdual_kkacc_1d",
    "tdual_virial_array": "tdual_kkacc_1d_6",
}


def patch_file(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    patched = text
    for old, new in REPLACEMENTS.items():
        patched = patched.replace(old, new)
    if patched != text:
        path.write_text(patched, encoding="utf-8")


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: patch_pair_nequip_allegro_lammps_10sep2025.py SOURCE_DIR", file=sys.stderr)
        return 2
    source_dir = Path(sys.argv[1])
    for name in ("pair_nequip_allegro_kokkos.h", "pair_nequip_allegro_kokkos.cpp"):
        path = source_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        patch_file(path)
    print("patched pair_nequip_allegro Kokkos typedefs for LAMMPS patch_10Sep2025")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
