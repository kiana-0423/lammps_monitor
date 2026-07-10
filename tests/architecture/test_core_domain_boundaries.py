"""Executable dependency rules for the PHAL core domain."""

from __future__ import annotations

import ast
from pathlib import Path

CORE_ROOT = Path(__file__).resolve().parents[2] / "src" / "hotspot_al"
CORE_PACKAGES = (
    "active_learning",
    "datasets",
    "detectors",
    "extraction",
    "hotspot",
    "workflows",
)

# Core algorithms may depend on stable domain models, NumPy, and ASE, but not
# on concrete scientific programs, process launchers, or plugin infrastructure.
FORBIDDEN_IMPORTS = (
    "allegro",
    "cp2k",
    "kubernetes",
    "lammps",
    "nequip",
    "subprocess",
    "torch",
    "hotspot_al.backends",
    "hotspot_al.cp2k",
    "hotspot_al.lammps",
    "hotspot_al.runtime",
    "hotspot_al.schedulers",
    "hotspot_al.training.allegro_runner",
)


def test_core_domain_does_not_import_infrastructure() -> None:
    violations: list[str] = []
    for package in CORE_PACKAGES:
        for path in sorted((CORE_ROOT / package).rglob("*.py")):
            for imported, line in _imports(path):
                if _is_forbidden(imported):
                    relative = path.relative_to(CORE_ROOT.parent)
                    violations.append(f"{relative}:{line} imports {imported}")

    assert violations == [], "Core Domain must not depend on Infrastructure:\n" + "\n".join(violations)


def _imports(path: Path) -> list[tuple[str, int]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend((alias.name, node.lineno) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.append((node.module, node.lineno))
    return imported


def _is_forbidden(imported: str) -> bool:
    return any(imported == prefix or imported.startswith(f"{prefix}.") for prefix in FORBIDDEN_IMPORTS)
