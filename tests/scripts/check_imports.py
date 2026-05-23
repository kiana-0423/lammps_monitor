"""Recursively import all hotspot_al modules."""

from __future__ import annotations

import importlib
import pkgutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def main() -> int:
    import hotspot_al

    failures: list[tuple[str, Exception]] = []
    for module in pkgutil.walk_packages(hotspot_al.__path__, hotspot_al.__name__ + "."):
        try:
            importlib.import_module(module.name)
        except Exception as exc:  # pragma: no cover - printed failure path
            failures.append((module.name, exc))

    if failures:
        for module_name, exc in failures:
            print(f"{module_name}: {type(exc).__name__}: {exc}")
        return 1

    print("All hotspot_al modules imported successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

