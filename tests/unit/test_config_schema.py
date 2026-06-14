"""Tests for configuration schema validation."""

from __future__ import annotations

import pytest

from hotspot_al.config import load_config
from hotspot_al.config_schema import validate_config
from hotspot_al.exceptions import ConfigError


def test_default_config_passes_schema_validation() -> None:
    config = load_config()

    assert config["project"]["name"] == "hotspot_active_learning"


def test_config_schema_reports_nested_type_errors() -> None:
    config = load_config()
    config["ood_score"]["lj_lazy_threshold"] = "too-high"

    with pytest.raises(ConfigError, match="ood_score.lj_lazy_threshold"):
        validate_config(config)


def test_config_schema_reports_invalid_ranges() -> None:
    config = load_config()
    config["online"]["monitor_freq"] = 0

    with pytest.raises(ConfigError, match="online.monitor_freq"):
        validate_config(config)


def test_config_schema_reports_unknown_nested_keys() -> None:
    config = load_config()
    config["cp2k"]["typo_cutoff"] = 500

    with pytest.raises(ConfigError, match="Unknown config key.*cp2k.*typo_cutoff"):
        validate_config(config)


def test_config_schema_validates_arbitrary_dict_values() -> None:
    config = load_config()
    config["ood_score"]["weights"]["force"] = "high"

    with pytest.raises(ConfigError, match="ood_score.weights.force"):
        validate_config(config)
