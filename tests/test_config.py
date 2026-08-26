"""Tests for configuration loading and validation."""
from pathlib import Path
import pytest
from src.config import load_config, Config


def test_load_config_returns_config_instance():
    """load_config should return a Config dataclass."""
    config = load_config(Path("config/indicators.yaml"))
    assert isinstance(config, Config)


def test_config_has_required_keys():
    """Config should contain features, target, and country settings."""
    config = load_config(Path("config/indicators.yaml"))
    assert len(config.features) > 0
    assert config.target_code is not None
    assert config.random_state == 42
    assert len(config.african_countries) >= 50


def test_african_countries_list_no_mena():
    """B7 FIX: Verify no Middle Eastern ISO3 codes in the list."""
    config = load_config(Path("config/indicators.yaml"))
    mena_codes = {"SAU", "IRN", "IRQ", "ISR", "JOR", "KWT", "OMN", "QAT",
                  "ARE", "YEM", "SYR", "LBN"}
    overlap = mena_codes.intersection(set(config.african_countries))
    assert len(overlap) == 0, f"MENA codes found in African list: {overlap}"