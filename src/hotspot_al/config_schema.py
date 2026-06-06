"""Lightweight schema validation for project configuration dictionaries."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from hotspot_al.exceptions import ConfigError


Schema = Mapping[str, Any]


CONFIG_SCHEMA: Schema = {
    "project": {
        "name": str,
        "method_name": str,
        "description": str,
    },
    "backend": {
        "mlip": str,
        "md_engine": str,
        "dft_engine": str,
    },
    "lammps": {
        "executable": str,
        "input_template": (str, type(None)),
        "dump_style": str,
        "dump_fields": list,
        "type_map": dict,
        "timestep_fs": (int, float),
    },
    "allegro": {
        "enabled": bool,
        "device": str,
        "model_paths": list,
        "deployed_model_paths": list,
        "dataset_dir": (str, type(None)),
        "train_output_dir": (str, type(None)),
        "checkpoint_path": (str, type(None)),
        "train_config_path": (str, type(None)),
        "train_command_template": (str, type(None)),
        "export_command_template": (str, type(None)),
        "lammps_pair_style": str,
        "committee_mode": str,
    },
    "monitor": {
        "light_interval": int,
        "physics_interval": (str, int),
        "committee_interval": (str, int),
        "force_z_threshold": (int, float),
        "delta_force_z_threshold": (int, float),
        "delta_q_threshold": (int, float),
        "rmin_threshold": (int, float, type(None)),
        "displacement_z_threshold": (int, float),
        "coordination_smoothing_power": (int, float),
        "coordination_scale": (int, float),
        "lj_cutoff": (int, float),
    },
    "online": {
        "enabled": bool,
        "work_dir": str,
        "dump_file": str,
        "dump_freq": int,
        "monitor_freq": int,
        "max_walltime": str,
        "event_dir": str,
    },
    "ood_score": {
        "weights": dict,
        "screen_threshold": (int, float),
        "physics_threshold": (int, float),
        "label_threshold": (int, float),
        "lj_lazy_threshold": (int, float),
        "min_trigger_atoms": int,
        "running_stats": {
            "enabled": bool,
            "warmup_frames": int,
            "min_std": (int, float),
        },
    },
    "buffer": {
        "pre_trigger_frames": int,
        "post_trigger_frames": int,
        "maxlen": (int, type(None)),
    },
    "hotspot": {
        "threshold": (int, float),
        "merge_radius": (int, float),
    },
    "extraction": {
        "mode": str,
        "mlip_cutoff": (int, float),
        "core_radius": (int, float),
        "buffer_radius": (int, float),
        "extract_radius": (int, float),
        "boundary_thickness": (int, float),
        "max_atoms": int,
        "min_atoms": int,
        "vacuum_padding": (int, float),
        "slab": dict,
        "graph": dict,
    },
    "h_capping": {
        "enabled": bool,
        "covalent_scale": (int, float),
        "allow_core_capping": bool,
        "optimize_h_only": bool,
        "disabled_for_metals": bool,
        "disabled_for_oxides_by_default": bool,
        "bond_lengths": dict,
    },
    "cp2k": {
        "executable": str,
        "functional": str,
        "basis": str,
        "potential": str,
        "cutoff": (int, float),
        "rel_cutoff": (int, float),
        "charge": int,
        "multiplicity": int,
        "poisson_solver": str,
        "periodic": str,
        "scf_eps": (int, float),
        "max_scf": int,
        "use_ot": bool,
        "dispersion": bool,
        "h_only_opt": dict,
        "single_point": dict,
    },
    "training_mask": dict,
    "candidate_pool": dict,
}


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate the known config schema and return ``config`` unchanged."""

    if not isinstance(config, dict):
        msg = f"config must be dict, got {type(config).__name__}"
        raise ConfigError(msg)
    _validate_mapping(config, CONFIG_SCHEMA, path="")
    _validate_ranges(config)
    return config


def _validate_mapping(config: Mapping[str, Any], schema: Schema, *, path: str) -> None:
    for key, expected in schema.items():
        dotted = f"{path}.{key}" if path else key
        if key not in config:
            msg = f"Missing required config key: {dotted}"
            raise ConfigError(msg)
        value = config[key]
        if isinstance(expected, Mapping):
            if not isinstance(value, Mapping):
                msg = f"{dotted} must be dict, got {type(value).__name__}"
                raise ConfigError(msg)
            _validate_mapping(value, expected, path=dotted)
        elif not _matches_type(value, expected):
            expected_name = _type_name(expected)
            msg = f"{dotted} must be {expected_name}, got {type(value).__name__}"
            raise ConfigError(msg)


def _matches_type(value: Any, expected: Any) -> bool:
    if isinstance(expected, tuple):
        return any(_matches_type(value, item) for item in expected)
    if expected is bool:
        return isinstance(value, bool)
    if expected is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if expected is float:
        return isinstance(value, float) and not isinstance(value, bool)
    if expected in (str, list, dict, type(None)):
        return isinstance(value, expected)
    if expected == (int, float):
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return isinstance(value, expected)


def _validate_ranges(config: Mapping[str, Any]) -> None:
    positive_paths = (
        ("lammps", "timestep_fs"),
        ("online", "dump_freq"),
        ("online", "monitor_freq"),
        ("buffer", "pre_trigger_frames"),
        ("buffer", "post_trigger_frames"),
        ("monitor", "lj_cutoff"),
        ("extraction", "extract_radius"),
        ("extraction", "max_atoms"),
        ("extraction", "min_atoms"),
        ("cp2k", "max_scf"),
    )
    for path in positive_paths:
        value = _get_path(config, path)
        if value is not None and value <= 0:
            msg = f"{'.'.join(path)} must be positive, got {value!r}"
            raise ConfigError(msg)


def _get_path(config: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = config
    for part in path:
        value = value[part]
    return value


def _type_name(expected: Any) -> str:
    if isinstance(expected, tuple):
        return " or ".join(_type_name(item) for item in expected)
    if expected is type(None):
        return "None"
    return expected.__name__
