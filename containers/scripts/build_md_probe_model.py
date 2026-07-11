"""Build a tiny random Allegro integration probe and compile it for LAMMPS.

The generated model is intentionally not trained and is not scientifically valid.
It exists only to verify the LAMMPS pair_allegro integration path.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import hydra
import torch
import yaml
from hydra.utils import instantiate
from lightning import Trainer
from omegaconf import OmegaConf

from nequip.data.datamodule import NequIPDataModule
from nequip.scripts.train import _REQUIRED_CONFIG_SECTIONS
from nequip.train import NequIPLightningModule
from nequip.utils.global_state import get_latest_global_state, set_global_state
from nequip.utils.versions import get_current_code_versions


PROBE_DIR = Path("/opt/phal/probes/md")
WORK_DIR = PROBE_DIR / "model-build"
CONFIG_PATH = WORK_DIR / "allegro-probe-config.yaml"
DATA_PATH = WORK_DIR / "probe.xyz"
CHECKPOINT_PATH = PROBE_DIR / "allegro-probe-initial.ckpt"
COMPILED_MODEL_PATH = PROBE_DIR / "allegro-probe.nequip.pth"
METADATA_PATH = PROBE_DIR / "allegro-probe-metadata.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_probe_inputs() -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PATH.write_text(
        """2
Properties=species:S:1:pos:R:3:forces:R:3 energy=0.0 pbc="F F F"
H 0.0 0.0 0.0 0.0 0.0 0.0
H 1.0 0.0 0.0 0.0 0.0 0.0
2
Properties=species:S:1:pos:R:3:forces:R:3 energy=0.0 pbc="F F F"
H 0.0 0.0 0.0 0.0 0.0 0.0
H 1.1 0.0 0.0 0.0 0.0 0.0
2
Properties=species:S:1:pos:R:3:forces:R:3 energy=0.0 pbc="F F F"
H 0.0 0.0 0.0 0.0 0.0 0.0
H 1.2 0.0 0.0 0.0 0.0 0.0
""",
        encoding="utf-8",
    )
    config = {
        "run": ["train"],
        "dataset_file_name": str(DATA_PATH),
        "cutoff_radius": 2.5,
        "chemical_symbols": ["H"],
        "model_type_names": ["H"],
        "seed": 20251103,
        "model_dtype": "float32",
        "data": {
            "_target_": "nequip.data.datamodule.ASEDataModule",
            "seed": "${seed}",
            "split_dataset": {"file_path": "${dataset_file_name}", "train": 1, "val": 1, "test": 1},
            "transforms": [
                {"_target_": "nequip.data.transforms.NeighborListTransform", "r_max": "${cutoff_radius}"},
                {
                    "_target_": "nequip.data.transforms.ChemicalSpeciesToAtomTypeMapper",
                    "model_type_names": "${model_type_names}",
                    "chemical_species_to_atom_type_map": {"H": "H"},
                },
            ],
            "train_dataloader": {"_target_": "torch.utils.data.DataLoader", "batch_size": 1},
            "val_dataloader": {"_target_": "torch.utils.data.DataLoader", "batch_size": 1},
            "test_dataloader": {"_target_": "torch.utils.data.DataLoader", "batch_size": 1},
            "stats_manager": {
                "_target_": "nequip.data.DataStatisticsManager",
                "dataloader_kwargs": {"batch_size": 1},
                "metrics": [
                    {
                        "field": {"_target_": "nequip.data.NumNeighbors"},
                        "metric": {"_target_": "nequip.data.Mean"},
                        "name": "num_neighbors_mean",
                    },
                    {
                        "field": {"_target_": "nequip.data.PerAtomModifier", "field": "total_energy"},
                        "metric": {"_target_": "nequip.data.Mean"},
                        "name": "per_atom_energy_mean",
                    },
                    {
                        "field": "forces",
                        "metric": {"_target_": "nequip.data.RootMeanSquare"},
                        "name": "forces_rms",
                    },
                ],
            },
        },
        "trainer": {"_target_": "lightning.Trainer", "max_epochs": 0, "logger": False},
        "training_module": {
            "_target_": "nequip.train.EMALightningModule",
            "loss": {
                "_target_": "nequip.train.EnergyForceLoss",
                "per_atom_energy": True,
                "coeffs": {"total_energy": 1.0, "forces": 1.0},
            },
            "val_metrics": {
                "_target_": "nequip.train.EnergyForceMetrics",
                "coeffs": {"total_energy_mae": 1.0, "forces_mae": 1.0},
            },
            "optimizer": {"_target_": "torch.optim.Adam", "lr": 0.004},
            "model": {
                "_target_": "allegro.model.AllegroModel",
                "seed": "${seed}",
                "model_dtype": "${model_dtype}",
                "type_names": "${model_type_names}",
                "r_max": "${cutoff_radius}",
                "radial_chemical_embed": {
                    "_target_": "allegro.nn.TwoBodyBesselScalarEmbed",
                    "num_bessels": 2,
                    "bessel_trainable": False,
                    "polynomial_cutoff_p": 6,
                    "module_output_dim": 4,
                },
                "l_max": 1,
                "parity": False,
                "num_layers": 1,
                "num_scalar_features": 4,
                "num_tensor_features": 2,
                "allegro_mlp_hidden_layers_depth": 1,
                "allegro_mlp_hidden_layers_width": 4,
                "readout_mlp_hidden_layers_depth": 1,
                "readout_mlp_hidden_layers_width": 4,
                "avg_num_neighbors": "${training_data_stats:num_neighbors_mean}",
                "per_type_energy_shifts": "${training_data_stats:per_atom_energy_mean}",
                "per_type_energy_scales": "${training_data_stats:forces_rms}",
            },
        },
        "global_options": {"allow_tf32": False},
        "metadata": {"purpose": "integration_probe_only", "scientifically_valid": False},
    }
    CONFIG_PATH.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def build_checkpoint() -> None:
    os.chdir(WORK_DIR)
    config = OmegaConf.load(CONFIG_PATH)
    for section in _REQUIRED_CONFIG_SECTIONS:
        if section not in config:
            raise KeyError(section)

    set_global_state()
    import nequip.utils.xpu  # noqa: F401

    data_config = OmegaConf.to_container(config.data, resolve=True)
    datamodule = instantiate(data_config, _recursive_=False)
    if not isinstance(datamodule, NequIPDataModule):
        raise TypeError(type(datamodule))
    stats = datamodule.get_statistics(dataset="train")

    def training_data_stats(stat_name: str):
        stat = stats.get(stat_name, None)
        if stat is None:
            raise RuntimeError(stat_name)
        return stat

    OmegaConf.register_new_resolver("training_data_stats", training_data_stats, replace=True, use_cache=True)
    module_config = OmegaConf.to_container(config.training_module, resolve=True)
    info = {
        "versions": get_current_code_versions(),
        "data": data_config,
        "trainer": {"_target_": "lightning.Trainer", "inference_mode": False},
        "global_options": get_latest_global_state(),
        "runs": ["train"],
        "training_module": module_config,
        "phal_probe": {"purpose": "integration_probe_only", "scientifically_valid": False},
    }
    module_cls = hydra.utils.get_class(config.training_module._target_)
    if not issubclass(module_cls, NequIPLightningModule):
        raise TypeError(module_cls)
    module = instantiate(
        module_config,
        _recursive_=False,
        _convert_="all",
        num_datasets=datamodule.num_datasets,
        info_dict=info,
    )
    trainer = Trainer(logger=False, enable_checkpointing=False, inference_mode=False)
    module.world_size = trainer.world_size
    trainer.strategy.connect(module)
    trainer.save_checkpoint(CHECKPOINT_PATH)


def compile_model() -> dict[str, object]:
    command = [
        "nequip-compile",
        "--mode",
        "torchscript",
        "--device",
        "cpu",
        "--target",
        "pair_allegro",
        str(CHECKPOINT_PATH),
        str(COMPILED_MODEL_PATH),
    ]
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    return {
        "compile_command": command,
        "compile_stdout": completed.stdout,
        "compile_stderr": completed.stderr,
    }


def main() -> None:
    torch.manual_seed(20251103)
    write_probe_inputs()
    build_checkpoint()
    compile_info = compile_model()
    metadata = {
        "purpose": "integration_probe_only",
        "scientifically_valid": False,
        "model_kind": "random_initialized_allegro_probe",
        "training_steps": 0,
        "type_names": ["H"],
        "checkpoint_path": str(CHECKPOINT_PATH),
        "compiled_model_path": str(COMPILED_MODEL_PATH),
        "compiled_model_format": "torchscript",
        "compiled_model_sha256": sha256(COMPILED_MODEL_PATH),
        "torch": torch.__version__,
        "torch_cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
        **compile_info,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
