# Configuration Reference

The default configuration lives in `config/default.yaml` and is validated by
`hotspot_al.config_schema.validate_config()` when loaded through
`hotspot_al.config.load_config()`.

Important sections:

- `backend`: selected MLIP, MD, and DFT engines.
- `lammps`: executable, dump fields, atom type map, timestep, input templates.
- `allegro`: model paths, deployed model paths, train/export command templates.
- `online`: work directory, dump file, dump frequency, monitor frequency.
- `ood_score`: metric weights, trigger thresholds, running-stat settings.
- `buffer`: pre/post trigger frame counts.
- `extraction`: local region size, core/buffer radii, max/min atoms.
- `h_capping`: conservative hydrogen capping controls.
- `cp2k`: CP2K executable, DFT settings, submit mode, retry settings.
- `training_mask`: atom-wise supervision weights by region label.
- `retraining`: optional automatic retraining trigger settings.
