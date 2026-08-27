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

UN_AFRICAN_MEMBER_STATES = {
    "DZA","AGO","BEN","BWA","BFA","BDI","CPV","CMR","CAF","TCD","COM","COG",
    "COD","CIV","DJI","EGY","GNQ","ERI","SWZ","ETH","GAB","GMB","GHA","GIN",
    "GNB","KEN","LSO","LBR","LBY","MDG","MWI","MLI","MRT","MUS","MAR","MOZ",
    "NAM","NER","NGA","RWA","STP","SEN","SYC","SLE","SOM","ZAF","SSD","SDN",
    "TZA","TGO","TUN","UGA","ZMB","ZWE",
}


def test_all_un_african_member_states_present():
    """M3: Mauritius and Sudan were silently absent. Assert full coverage."""
    config = load_config(Path("config/indicators.yaml"))
    missing = UN_AFRICAN_MEMBER_STATES - set(config.african_countries)
    assert not missing, f"Missing UN African member states: {sorted(missing)}"


def test_country_list_has_no_unexpected_entries():
    """Only ESH (Western Sahara) may sit outside the UN member list."""
    config = load_config(Path("config/indicators.yaml"))
    extra = set(config.african_countries) - UN_AFRICAN_MEMBER_STATES
    assert extra <= {"ESH"}, f"Unexpected non-member entries: {sorted(extra)}"


def test_country_list_has_no_duplicates():
    """55 entries, each unique (54 UN member states + ESH)."""
    config = load_config(Path("config/indicators.yaml"))
    assert len(config.african_countries) == 55
    assert len(set(config.african_countries)) == 55


@pytest.mark.xfail(reason="Committed panel predates MUS/SDN; see data/README.md "
                          "Known Data Limitations")
def test_panel_covers_all_configured_countries():
    """Expected to fail on the committed 52-country panel (Task 1.4 Path B)."""
    import pandas as pd
    panel = pd.read_parquet("data/processed/model_data.parquet")
    cfg = load_config(Path("config/indicators.yaml"))
    expected = set(cfg.african_countries) - {"ESH"}
    assert expected <= set(panel["iso3"].unique())
