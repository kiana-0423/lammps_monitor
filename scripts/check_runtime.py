"""Check whether the external Hotspot-AL runtime is available."""

from __future__ import annotations

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = PROJECT_ROOT / "src" / "hotspot_al" / "cli.py"


def _load_main():
    spec = spec_from_file_location("hotspot_al_runtime_cli", CLI_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load runtime checker: {CLI_PATH}")
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.main


if __name__ == "__main__":
    main = _load_main()
    raise SystemExit(main(sys.argv[1:]))
