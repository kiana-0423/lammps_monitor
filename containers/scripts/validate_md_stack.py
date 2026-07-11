"""Validate the PHAL md-probe container without requiring a local NVIDIA GPU."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import subprocess
import traceback
from pathlib import Path
from typing import Any


PROBE_DIR = Path("/opt/phal/probes/md")
MODEL_PATH = PROBE_DIR / "allegro-probe.nequip.pth"
RUN0_INPUT = PROBE_DIR / "in.run0"
RUN10_INPUT = PROBE_DIR / "in.run10"


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_command(command: list[str], cwd: Path | None = None) -> dict[str, Any]:
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def parse_lammps_version(help_text: str) -> str | None:
    match = re.search(r"LAMMPS \\(([^)]+)\\)", help_text)
    return match.group(1) if match else None


def thermo_is_finite(output: str) -> bool:
    lines = output.splitlines()
    header_index = None
    for index, line in enumerate(lines):
        if re.match(r"\\s*Step\\s+Temp\\s+PotEng\\s+KinEng\\s+TotEng\\s*$", line):
            header_index = index
            break
    if header_index is None:
        return False
    found = False
    for line in lines[header_index + 1 :]:
        stripped = line.strip()
        if not stripped or stripped.startswith("Loop time"):
            break
        parts = stripped.split()
        if len(parts) != 5:
            continue
        try:
            values = [float(value) for value in parts]
        except ValueError:
            continue
        if not all(math.isfinite(value) for value in values):
            return False
        found = True
    return found


def main() -> int:
    result: dict[str, Any] = {
        "platform.machine": platform.machine(),
        "compiled_model_path": str(MODEL_PATH),
        "compiled_model_sha256": sha256(MODEL_PATH),
        "lmp_help_passed": False,
        "pair_allegro_available": False,
        "run0_passed": False,
        "run10_passed": False,
        "thermo_finite": False,
        "cpu_status": "not_run",
        "cuda_status": "skipped",
        "overall_status": "FAIL",
        "stderr": "",
        "traceback": "",
    }
    try:
        import allegro
        import nequip
        import torch

        result.update(
            {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "torch.cuda.is_available": torch.cuda.is_available(),
                "torch_cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
                "nequip": nequip.__version__,
                "allegro": allegro.__version__,
            }
        )

        help_run = run_command(["lmp", "-h"])
        result["lmp_help"] = help_run
        result["lmp_help_passed"] = help_run["returncode"] == 0
        help_text = help_run["stdout"] + help_run["stderr"]
        result["LAMMPS version"] = parse_lammps_version(help_text)
        result["pair_allegro_available"] = "allegro" in help_text
        result["pair_allegro_kokkos_available"] = "allegro/kk" in help_text

        run0 = run_command(["lmp", "-in", str(RUN0_INPUT)], cwd=PROBE_DIR)
        result["run0"] = run0
        result["run0_passed"] = run0["returncode"] == 0

        run10 = run_command(["lmp", "-in", str(RUN10_INPUT)], cwd=PROBE_DIR)
        result["run10"] = run10
        result["run10_passed"] = run10["returncode"] == 0
        result["thermo_finite"] = thermo_is_finite(run10["stdout"] + run10["stderr"])
        result["cpu_status"] = "passed" if result["run0_passed"] and result["run10_passed"] else "failed"

        if (
            result["lmp_help_passed"]
            and result["pair_allegro_available"]
            and result["compiled_model_sha256"]
            and result["run0_passed"]
            and result["run10_passed"]
            and result["thermo_finite"]
        ):
            result["overall_status"] = "PASS"
    except Exception as exc:  # pragma: no cover - container diagnostic path
        result["stderr"] = str(exc)
        result["traceback"] = traceback.format_exc()

    print(json.dumps(result, indent=2))
    return 0 if result["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
