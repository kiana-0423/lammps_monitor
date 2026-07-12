# Configuration Reference

The default configuration lives in `config/default.yaml` and is validated by
`hotspot_al.config_schema.validate_config()` when loaded through
`hotspot_al.config.load_config()`.

Important sections:

- `backend`: nested `md`, `mlip`, `dft`, and `scheduler` engine selections.
- `datasets`: backend-neutral labeled and training dataset locations.
- `plugins`: free-form namespaced settings owned by third-party backends.
- `lammps`: executable, dump fields, atom type map, timestep, input templates.
- `allegro`: model paths, deployed model paths, train/export command templates.
- `allegro_model_input`: optional extra model-input fields passed to
  `AllegroInference`.
- `online`: work directory, dump file, dump frequency, monitor frequency,
  optional progress JSON file, and error-continuation controls.
- `monitor`: online stage cadence and per-metric thresholds. `light_interval`,
  `physics_interval`, and `committee_interval` accept positive integers for
  periodic sampling. The string `triggered` means the stage runs only when the
  previous stage triggers. `online.monitor_freq > 1` adds a periodic forced-full
  sample on matching frame indices; the default `monitor_freq: 1` follows the
  staged cascade instead of forcing every frame to full.
- `logging`: text or JSON logs via `logging.format`, plus optional file output.
- `ood_score`: metric weights, trigger thresholds, running-stat settings.
- `buffer`: pre/post trigger frame counts.
- `extraction`: local region size, core/buffer radii, max/min atoms.
- `extraction.block`: block-mode settings used when `extraction.mode=block`.
  Current implementation supports `scheme: spatial_grid`, maps OOD atoms to
  stable block ids, optionally merges 26-neighbor-connected blocks, and applies
  cooldown before re-submitting a block for DFT labeling.
  - `size`: three block dimensions in Angstrom, for example `[12.0, 12.0, 12.0]`.
  - `halo`: context distance beyond the block core.
  - `merge_adjacent`: merge adjacent anomalous blocks before extraction.
  - `max_merged_blocks`: cap merged block group size.
  - `cooldown_steps`: minimum MD steps before the same block can be labeled again.
  - `buffer.inner`: distance from core atoms to inner-buffer outer edge.
  - `buffer.outer`: distance from core atoms to outer-buffer outer edge.
  - `frozen.enabled`: include a frozen boundary shell for CP2K constraints.
  - `frozen.thickness`: frozen shell thickness.
  - `max_atoms` / `min_atoms`: atom-count bounds for extracted block regions.
- `h_capping`: conservative hydrogen capping controls.
- `cp2k`: CP2K executable, DFT settings, submit mode, retry settings.
- `training_mask`: atom-wise supervision weights by region label. Block mode
  accepts `label_core` and `frozen_boundary` aliases alongside the existing
  `core` and `boundary` keys.
- `retraining`: optional automatic retraining trigger settings.

`validate_config()` checks required keys, nested value types, positive ranges,
and unknown keys. Extra site-local settings should live outside the PHAL config
or be added to `CONFIG_SCHEMA` before being loaded through `load_config()`.
