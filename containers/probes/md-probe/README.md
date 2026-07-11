# PHAL md-probe

This probe validates LAMMPS `pair_style allegro` integration only.

The included model is generated during the container build from a fixed random
seed, compiled with `nequip-compile`, and marked:

- `purpose = integration_probe_only`
- `scientifically_valid = false`

The `run 10` input is a CPU smoke test and is not a formal MD simulation.
