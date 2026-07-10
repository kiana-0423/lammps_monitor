# Runtime Data Contract

All generated operational data lives below the repository-local `runtime/`
root during development and is mounted at `/runtime` inside containers.

```text
runtime/
├── datasets/
├── models/
│   ├── latest/
│   └── archive/
├── checkpoints/
├── md/
│   ├── traj/
│   └── restart/
├── cp2k/
│   ├── inputs/
│   └── outputs/
├── logs/
└── cache/
```

The directory skeleton is tracked with empty placeholders. Actual files are
ignored by Git. Containers must access these locations through YAML-configured
paths and mounts; source modules must not assume a workstation or cluster path.

Models, trajectories, checkpoints, datasets, logs, caches, SIF files, and OCI
archives are artifacts, not source code.
