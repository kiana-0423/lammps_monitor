"""Command line helpers for runtime setup and diagnostics."""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


DEFAULT_RUNTIME_CONFIG = Path("config") / "runtime.local.yaml"
LMP_CANDIDATES = ("lmp_allegro", "lmp", "lammps")
CP2K_CANDIDATES = ("cp2k.popt", "cp2k.psmp", "cp2k")
PYTHON_MODULES = ("torch", "nequip", "allegro", "ase")


@dataclass(frozen=True, slots=True)
class CheckResult:
    """A single runtime check result."""

    name: str
    ok: bool
    detail: str


def _find_executable(candidates: Sequence[str]) -> str | None:
    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return None


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _torch_cuda_detail() -> CheckResult:
    if not _module_available("torch"):
        return CheckResult("cuda", False, "torch is not importable")
    try:
        import torch
    except Exception as exc:  # pragma: no cover - defensive around binary imports
        return CheckResult("cuda", False, f"torch import failed: {exc}")

    version = getattr(torch, "__version__", "unknown")
    cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    available = bool(torch.cuda.is_available())
    if available:
        device_name = torch.cuda.get_device_name(0)
        return CheckResult("cuda", True, f"torch {version}, cuda {cuda_version}, device {device_name}")
    return CheckResult("cuda", False, f"torch {version}, cuda {cuda_version}, cuda unavailable")


def collect_runtime_checks(
    *,
    lammps_candidates: Sequence[str] = LMP_CANDIDATES,
    cp2k_candidates: Sequence[str] = CP2K_CANDIDATES,
    modules: Sequence[str] = PYTHON_MODULES,
) -> tuple[list[CheckResult], dict[str, str]]:
    """Check external executables and optional Python runtimes."""

    paths: dict[str, str] = {}
    results: list[CheckResult] = []

    lammps_bin = os.environ.get("LAMMPS_BIN") or _find_executable(lammps_candidates)
    if lammps_bin:
        paths["lammps"] = lammps_bin
        results.append(CheckResult("lammps", True, lammps_bin))
    else:
        results.append(CheckResult("lammps", False, "not found; set LAMMPS_BIN or add lmp_allegro to PATH"))

    cp2k_bin = os.environ.get("CP2K_BIN") or _find_executable(cp2k_candidates)
    if cp2k_bin:
        paths["cp2k"] = cp2k_bin
        results.append(CheckResult("cp2k", True, cp2k_bin))
    else:
        results.append(CheckResult("cp2k", False, "not found; set CP2K_BIN or add cp2k.popt to PATH"))

    for module in modules:
        ok = _module_available(module)
        detail = "importable" if ok else "not importable"
        results.append(CheckResult(module, ok, detail))

    results.append(_torch_cuda_detail())
    return results, paths


def write_runtime_config(path: str | Path, executable_paths: dict[str, str]) -> Path:
    """Write a config override pointing at discovered runtime executables."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    if "lammps" in executable_paths:
        lines.extend(["lammps:", f"  executable: {executable_paths['lammps']}"])
    if "cp2k" in executable_paths:
        lines.extend(["cp2k:", f"  executable: {executable_paths['cp2k']}"])
    with target.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
        handle.write("\n")
    return target


def _format_result(result: CheckResult) -> str:
    status = "OK" if result.ok else "MISS"
    return f"[{status}] {result.name}: {result.detail}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="check_runtime.py", description="Hotspot-AL runtime diagnostics.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Check LAMMPS/CP2K/Allegro runtime availability.")
    doctor.add_argument(
        "--write-config",
        nargs="?",
        const=str(DEFAULT_RUNTIME_CONFIG),
        default=None,
        help="Write a local config override with discovered executable paths.",
    )
    doctor.add_argument(
        "--strict",
        action="store_true",
        help="Return a non-zero status if any runtime check is missing.",
    )
    return parser


def run_doctor(args: argparse.Namespace) -> int:
    results, paths = collect_runtime_checks()
    for result in results:
        print(_format_result(result))

    if args.write_config is not None:
        target = write_runtime_config(args.write_config, paths)
        print(f"wrote runtime config: {target}")

    if args.strict and any(not result.ok for result in results):
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "doctor":
        return run_doctor(args)
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
