"""Allegro backend hooks for LAMMPS active learning workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hotspot_al.lammps.backend import MLIPBackend
from hotspot_al.lammps.lammps_input import build_lammps_input
from hotspot_al.training.allegro_adapter import write_allegro_dataset


class AllegroBackend(MLIPBackend):
    """Allegro backend adapter."""

    name = "allegro"

    def write_lammps_input(self, config: dict[str, Any]) -> str:
        allegro_cfg = config.get("allegro", {})
        model_paths = allegro_cfg.get("deployed_model_paths") or allegro_cfg.get("model_paths", [])
        if not model_paths:
            msg = "Allegro backend requires at least one model path."
            raise ValueError(msg)
        pair_style = f"pair_style {allegro_cfg.get('lammps_pair_style', 'allegro')} {model_paths[0]}"
        pair_coeff = "pair_coeff * *"
        return build_lammps_input("\n".join([pair_style, pair_coeff]), config=config)

    def write_training_data(self, region, *, forces, output_dir, config, filename: str = "dataset.extxyz"):
        return write_allegro_dataset(region, forces=forces, output_dir=output_dir, config=config, filename=filename)

    def export_model(self, output_dir: str | Path):
        return Path(output_dir)
