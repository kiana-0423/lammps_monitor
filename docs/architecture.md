# Extensible Architecture

PHAL is a platform for physics-aware hotspot active learning. Its domain code
does not select or launch Allegro, LAMMPS, CP2K, or a batch system directly.
Those responsibilities belong to backend plugins.

## First Principle

Physics-aware Active Learning is PHAL's only Core Domain. CP2K, LAMMPS,
Allegro, schedulers, containers, and site runtimes are Infrastructure. They are
replaceable tools and must never define the domain model or algorithm design.

This produces four layers:

| Layer | Responsibility | Allowed dependencies |
| --- | --- | --- |
| Core Domain | OOD, hotspot detection, extraction, candidate policy, dataset semantics | Domain models and controlled numerical/atomistic foundations |
| Application | Use cases, workflow coordination, retries and lifecycle | Core Domain and Backend ports |
| Ports | Stable MLIP/MD/DFT/Scheduler contracts | Core data types |
| Infrastructure | Allegro/LAMMPS/CP2K, Local/Slurm/PBS, containers | Ports, vendor APIs and operating-system runtimes |

Dependencies point inward: Infrastructure implements Ports; Application
orchestrates Ports and Core Domain; Core Domain never imports Infrastructure.
Composition roots may know all layers solely to construct and inject objects.

"No external dependencies" at the algorithm boundary means no dependency on a
specific scientific program, model runtime, scheduler, executable, or site
environment. NumPy and ASE are currently controlled foundations for numerical
arrays and atomistic data; they do not select an Infrastructure backend.

The rule is enforced by
`tests/architecture/test_core_domain_boundaries.py`. A concrete backend import
inside `active_learning`, `datasets`, `detectors`, `extraction`, `hotspot`, or
`workflows` fails the test suite.

## Dependency direction

```text
detectors / datasets / active-learning policy
                    |
                    v
               workflows
                    |
                    v
      MLIP / MD / DFT / Scheduler contracts
                    |
                    v
     built-in or third-party backend plugins
```

The stable contracts are defined in `hotspot_al.backends.base`:

- `MLIPBackend`: force inference, committees, training, export, model reload;
- `MDBackend`: MD execution requests and frame reading;
- `DFTBackend`: labeling inputs, execution requests, completion checks, parsing;
- `SchedulerBackend`: submit, poll, and cancel portable execution requests.

External commands are represented by `ExecutionRequest`. This prevents a DFT
or MD adapter from embedding Slurm/PBS/Kubernetes policy and lets the same
scientific backend run locally or on a different scheduler.

## Configuration-driven selection

```yaml
backend:
  md:
    engine: lammps
  mlip:
    engine: allegro
  dft:
    engine: cp2k
  scheduler:
    engine: slurm
```

Engine-specific settings remain in their own sections. Executable paths and
command templates must be configured there; algorithms must not discover or
hard-code them.

External packages store free-form, namespaced settings under `plugins`, for
example `plugins.mace`. This keeps the core schema strict without requiring a
PHAL release whenever a plugin adds an option.

## Adding a plugin

A plugin implements one contract and publishes a factory through a Python
entry point. For example, a MACE package can declare:

```toml
[project.entry-points."hotspot_al.backends"]
"mlip:mace" = "phal_mace.backend:MACEBackend"
```

`MACEBackend` may be either a `Backend` subclass with `from_config()`, or the
entry point may expose a factory accepting the complete configuration. PHAL
discovers the entry point on first use. Selecting it then requires only:

```yaml
backend:
  mlip:
    engine: mace
plugins:
  mace:
    model_paths: [models/mace.model]
```

The active-learning, detector, dataset, and workflow packages do not change.

Plugins can also be registered programmatically for site-local integrations:

```python
registry.register("dft", "vasp", VASPBackend.from_config)
backend = registry.create("dft", "vasp", config)
```

## Compatibility policy

The former `AllegroRunner` injection path and flat backend selection are
temporarily accepted at compatibility boundaries. New platform code should
inject `MLIPBackend`, `DFTBackend`, or `SchedulerBackend` instead. Compatibility
adapters must not leak into detector or dataset logic.

Built-in support currently includes Allegro, LAMMPS, CP2K, Local, Slurm, and
PBS adapters. MACE, NequIP, DeepMD, CHGNet, SevenNet, VASP, Quantum ESPRESSO,
ORCA, OpenMM, and Kubernetes belong in independent plugins implementing these
same contracts.
