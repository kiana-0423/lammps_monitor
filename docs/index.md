# hotspot_al Docs

`hotspot_al` implements a hotspot-localized active-learning loop for
Allegro-LAMMPS-CP2K workflows.

## Quick Start

```bash
pip install -e ".[dev]"
python -m pytest -q tests/unit tests/e2e -m "not integration and not slow"
python examples/06_online_monitor.py
```

The online example uses fake Allegro forces and CP2K dry-run submission by
default. Replace the injected frame source with `LAMMPSController` and configure
`cp2k.submit_mode` for real labeling.

## Core Concepts

- `FrameData`: one MD frame with ASE atoms, forces, velocities, and metadata.
- `OODFrameResult`: atom-wise metric scores and trigger state.
- `EventRecord`: buffered pre/trigger/post frames around an OOD event.
- `ExtractedRegion`: local cluster/slab/graph region with mask labels.
- `ScheduledTask`: active-learning work item generated from an event.

## Main Modules

- `hotspot_al.monitor`: online/offline OOD metrics and scoring.
- `hotspot_al.lammps`: LAMMPS input generation, launch hooks, dump streaming.
- `hotspot_al.cp2k`: CP2K input, force parsing, and event task submission.
- `hotspot_al.training`: dataset writing, retraining triggers, model registry.
- `hotspot_al.active_learning`: scheduling and candidate bookkeeping.

## References

- [Configuration Reference](config-reference.md)
- [Online Mode](online-mode.md)
- [Performance Benchmarks](performance.md)
