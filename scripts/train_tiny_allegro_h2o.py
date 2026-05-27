#!/usr/bin/env python
"""Train a tiny H/O NequIP fallback model for H2O smoke-test inference."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys


DATASET_PATH = Path("tests/fixtures/allegro/tiny_h2o_synthetic.extxyz")
MODEL_DIR = Path("tests/fixtures/models")
TRAIN_DIR = MODEL_DIR / "tiny_h2o_train"
CONFIG_PATH = TRAIN_DIR / "config.yaml"
RUN_DIR = TRAIN_DIR / "run"
MODEL_PATH = MODEL_DIR / "tiny_h2o_allegro_model.pth"


CONFIG_TEMPLATE = """\
run: [train]

seed: 7
cutoff_radius: 3.0
model_type_names: [O, H]
chemical_species: ${{model_type_names}}
per_edge_type_cutoff: null

dataloader:
  _target_: torch.utils.data.DataLoader
  batch_size: 2

data:
  _target_: nequip.data.datamodule.NequIPDataModule
  seed: ${{seed}}
  split_dataset:
    dataset:
      _target_: nequip.data.dataset.ASEDataset
      file_path: {dataset_path}
      transforms:
        - _target_: nequip.data.transforms.ChemicalSpeciesToAtomTypeMapper
          model_type_names: ${{model_type_names}}
          chemical_species_to_atom_type_map: ${{list_to_identity_dict:${{chemical_species}}}}
        - _target_: nequip.data.transforms.NeighborListTransform
          r_max: ${{cutoff_radius}}
    train: 8
    val: 2
    test: 2
  train_dataloader: ${{dataloader}}
  val_dataloader: ${{dataloader}}
  test_dataloader: ${{dataloader}}
  stats_manager:
    _target_: nequip.data.CommonDataStatisticsManager
    type_names: ${{model_type_names}}

trainer:
  _target_: lightning.Trainer
  max_epochs: 2
  accelerator: cpu
  devices: 1
  check_val_every_n_epoch: 1
  log_every_n_steps: 1
  enable_progress_bar: false
  callbacks:
    - _target_: lightning.pytorch.callbacks.ModelCheckpoint
      monitor: val0_epoch/weighted_sum
      dirpath: ${{hydra:runtime.output_dir}}
      filename: best
      save_last: true

training_module:
  _target_: nequip.train.EMALightningModule
  loss:
    _target_: nequip.train.EnergyForceLoss
    per_atom_energy: true
    coeffs:
      total_energy: 1.0
      forces: 1.0
  val_metrics:
    _target_: nequip.train.EnergyForceMetrics
    coeffs:
      total_energy_mae: 1.0
      forces_mae: 1.0
  train_metrics: ${{training_module.val_metrics}}
  test_metrics: ${{training_module.val_metrics}}
  optimizer:
    _target_: torch.optim.Adam
    lr: 0.01
  model:
    _target_: nequip.model.NequIPGNNModel
    seed: ${{seed}}
    model_dtype: float32
    type_names: ${{model_type_names}}
    r_max: ${{cutoff_radius}}
    l_max: 1
    parity: true
    num_layers: 1
    num_features: 4
    radial_mlp_depth: 1
    radial_mlp_width: 4
    avg_num_neighbors: ${{training_data_stats:per_type_num_neighbors_mean}}
    per_type_energy_shifts: ${{training_data_stats:per_atom_energy_mean}}
    per_type_energy_scales: ${{training_data_stats:per_type_forces_rms}}
"""


def main() -> int:
    if os.environ.get("CONDA_DEFAULT_ENV") != "allegro-mac":
        print("error: this training smoke test must run inside conda environment allegro-mac")
        return 4
    if not DATASET_PATH.is_file():
        print(f"error: dataset missing: {DATASET_PATH}")
        print("run: python scripts/make_tiny_h2o_dataset.py")
        return 2

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        CONFIG_TEMPLATE.format(dataset_path=DATASET_PATH.resolve()),
        encoding="utf-8",
    )

    command = [
        "nequip-train",
        "--config-path",
        str(TRAIN_DIR.resolve()),
        "--config-name",
        "config",
        f"hydra.run.dir={RUN_DIR.resolve()}",
        "hydra.job.chdir=false",
    ]
    print("training command:", " ".join(command))
    print("note: tiny NequIP fallback model for smoke testing only; no chemical accuracy")
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=240,
        )
    except subprocess.TimeoutExpired:
        print("error: tiny model training timed out")
        return 3

    print("nequip-train stdout:")
    print(result.stdout)
    print("nequip-train stderr:")
    print(result.stderr)
    if result.returncode != 0:
        print(f"error: nequip-train failed with exit code {result.returncode}")
        return result.returncode

    best_checkpoint = RUN_DIR / "best.ckpt"
    if not best_checkpoint.is_file():
        print(f"error: expected checkpoint not found: {best_checkpoint}")
        return 5
    shutil.copy2(best_checkpoint, MODEL_PATH)
    print(f"tiny H/O model checkpoint: {MODEL_PATH}")
    print("model kind: NequIP tiny fallback checkpoint, named for Allegro/NequIP smoke-test fixture")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
