"""Offline E2E protocol test using fixtures and fake backends only."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ase.io import read

from fake_backends.fake_allegro import fake_committee_evaluator
from fake_backends.fake_cp2k import write_fake_cp2k_force_output
from hotspot_al.config import load_config
from hotspot_al.cp2k.cp2k_force_parser import parse_cp2k_forces
from hotspot_al.cp2k.cp2k_input import write_cp2k_inputs
from hotspot_al.extraction.cluster_extractor import extract_cluster_region
from hotspot_al.extraction.h_capping import add_h_caps
from hotspot_al.io.lammps_reader import read_dump
from hotspot_al.monitor.force_monitor import delta_force_norms, force_norms
from hotspot_al.monitor.geometry_monitor import minimum_neighbor_distances
from hotspot_al.monitor.ood_score import OODScorer
from hotspot_al.training.allegro_adapter import write_allegro_dataset
from hotspot_al.training.mask_generator import generate_atom_mask


def test_offline_active_learning_pipeline(fixtures_dir: Path, tmp_path: Path) -> None:
    config = load_config()
    config["extraction"] = {
        **config["extraction"],
        "extract_radius": 4.0,
        "core_radius": 1.0,
        "boundary_thickness": 1.0,
        "min_atoms": 4,
        "max_atoms": 20,
        "vacuum_padding": 4.0,
    }

    frames = read_dump(
        fixtures_dir / "lammps" / "toy_hotspot.lammpstrj",
        type_map={1: "C", 3: "H"},
        timestep_fs=config["lammps"]["timestep_fs"],
    )
    frame = frames[0]
    assert frame.forces is not None

    force_metric = force_norms(frame.forces)
    committee_metric = np.zeros(len(frame.atoms))
    committee_metric[int(np.argmax(force_metric))] = 10.0
    metrics = {
        "force": force_metric,
        "delta_force": delta_force_norms(frame.forces, None),
        "rmin": minimum_neighbor_distances(frame.atoms),
        "committee": committee_metric,
    }
    result = OODScorer(config).score_full(metrics, update_stats=False, metadata={"event_id": "offline-e2e", "backend": "fake"})
    assert result.hotspot_indices

    region = extract_cluster_region(frame.atoms, result.hotspot_indices, config=config)
    assert 1 <= len(region.atoms) <= len(frame.atoms)

    capped_region = add_h_caps(frame.atoms, region, config=config)
    assert len(capped_region.atoms) >= len(region.atoms)

    mask_weights = generate_atom_mask(capped_region, config)
    capped_region.mask_weights = mask_weights
    assert len(mask_weights) == len(capped_region.atoms)

    cp2k_inputs = write_cp2k_inputs(capped_region, tmp_path / "cp2k", config=config, job_name="offline_e2e")
    assert cp2k_inputs["single_point_input"].is_file()

    fake_cp2k_output = write_fake_cp2k_force_output(tmp_path / "cp2k" / "offline_e2e.out", capped_region.atoms)
    forces = parse_cp2k_forces(fake_cp2k_output)
    assert forces.shape == (len(capped_region.atoms), 3)

    written = write_allegro_dataset(capped_region, forces=forces, output_dir=tmp_path / "allegro", config=config)
    assert written["dataset"].is_file()
    roundtrip_atoms = read(written["dataset"], format="extxyz")
    assert len(roundtrip_atoms) == len(capped_region.atoms)
    assert "mask_weights" in roundtrip_atoms.arrays

    committee_forces = fake_committee_evaluator(capped_region.atoms, ["model-a.pth", "model-b.pth"], config)
    assert committee_forces.shape == (2, len(capped_region.atoms), 3)

