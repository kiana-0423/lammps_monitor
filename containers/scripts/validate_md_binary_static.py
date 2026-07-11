"""Static md-probe binary validation for local Docker builders.

This script intentionally does not execute ``lmp``. Apple Silicon Docker
builders can compile and link the CUDA/Kokkos binary but do not provide the
NVIDIA host driver library ``libcuda.so.1`` at local build time.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path


LMP = Path("/opt/lammps/bin/lmp")
LAMMPS_HOME = Path("/opt/lammps")
LAMMPS_SOURCE = Path("/opt/src/lammps")
PAIR_SOURCE = Path("/opt/src/pair_nequip_allegro")


def run(command: list[str]) -> dict[str, object]:
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def main() -> int:
    lmp_exists = LMP.is_file()
    mode = LMP.stat().st_mode if lmp_exists else 0
    lmp_executable = bool(lmp_exists and mode & stat.S_IXUSR)
    file_result = run(["file", str(LMP)]) if lmp_exists else {"stdout": "", "stderr": "", "returncode": 127}
    ldd_result = run(["ldd", str(LMP)]) if lmp_exists else {"stdout": "", "stderr": "", "returncode": 127}
    file_text = str(file_result["stdout"] + file_result["stderr"])
    ldd_text = str(ldd_result["stdout"] + ldd_result["stderr"])

    pair_artifacts = {
        "upstream_source": (PAIR_SOURCE / "pair_nequip_allegro.cpp").is_file(),
        "upstream_kokkos_source": (PAIR_SOURCE / "pair_nequip_allegro_kokkos.cpp").is_file(),
        "installed_source": (LAMMPS_SOURCE / "src" / "pair_nequip_allegro.cpp").is_file(),
        "installed_kokkos_source": (LAMMPS_SOURCE / "src" / "KOKKOS" / "pair_nequip_allegro_kokkos.cpp").is_file(),
        "object": (LAMMPS_SOURCE / "build" / "CMakeFiles" / "lammps.dir" / "opt" / "src" / "lammps" / "src" / "pair_nequip_allegro.cpp.o").is_file(),
        "kokkos_object": (LAMMPS_SOURCE / "build" / "CMakeFiles" / "lammps.dir" / "opt" / "src" / "lammps" / "src" / "KOKKOS" / "pair_nequip_allegro_kokkos.cpp.o").is_file(),
    }

    result = {
        "lmp_path": str(LMP),
        "lmp_exists": lmp_exists,
        "lmp_executable": lmp_executable,
        "architecture": "aarch64" if "aarch64" in file_text.lower() or "arm aarch64" in file_text.lower() else "unknown",
        "file": file_result,
        "ldd": ldd_result,
        "lammps_home_exists": LAMMPS_HOME.is_dir(),
        "libtorch_dependency": "libtorch" in ldd_text,
        "cuda_driver_library": "not_available_on_local_builder" if "libcuda.so.1 => not found" in ldd_text else "available_or_not_linked",
        "pair_allegro_artifacts": pair_artifacts,
        "pair_allegro_compiled": pair_artifacts["object"] and pair_artifacts["kokkos_object"],
        "lmp_runtime": "not_run",
        "run0": "not_run",
        "run10": "not_run",
    }
    result["static_validation"] = (
        "passed"
        if all(
            (
                result["lmp_exists"],
                result["lmp_executable"],
                result["architecture"] == "aarch64",
                result["lammps_home_exists"],
                result["libtorch_dependency"],
                result["cuda_driver_library"] == "not_available_on_local_builder",
                result["pair_allegro_compiled"],
            )
        )
        else "failed"
    )

    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0 if result["static_validation"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
