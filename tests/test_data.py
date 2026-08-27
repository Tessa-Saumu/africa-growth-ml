"""Tests for data loading, filtering, and reshaping."""
import pandas as pd
import numpy as np
import pytest
from pathlib import Path
from src.data import (
    load_wdi_csv,
    check_duplicates,
    filter_african_countries,
    filter_indicators,
    reshape_wide_to_long,
    pivot_to_country_year,
    clean_numeric,
    log_dataset_info,
)


@pytest.fixture
def sample_wdi_df():
    """Create a small sample WDI dataframe for testing."""
    data = {
        "Country Name": ["Ghana", "Kenya", "Nigeria", "Sub-Saharan Africa"],
        "Country Code": ["GHA", "KEN", "NGA", "SSF"],
        "Indicator Name": ["GDP per capita growth", "GDP per capita growth", "GDP per capita growth", "GDP per capita growth"],
        "Indicator Code": ["NY.GDP.PCAP.KD.ZG", "NY.GDP.PCAP.KD.ZG", "NY.GDP.PCAP.KD.ZG", "NY.GDP.PCAP.KD.ZG"],
        "2018": [6.2, 5.1, 2.0, 3.0],
        "2019": [7.1, 5.4, 2.2, 3.5],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_metadata():
    """Create a small sample metadata dataframe."""
    return pd.DataFrame({
        "CountryCode": ["GHA", "KEN", "NGA", "SSF"],
        "Region": ["Sub-Saharan Africa", "Sub-Saharan Africa", "Sub-Saharan Africa", "Sub-Saharan Africa"],
    })


def test_filter_african_countries_with_iso3_list(sample_wdi_df):
    """B7 FIX: Using explicit ISO3 list (no more metadata/region substring)."""
    result = filter_african_countries(sample_wdi_df, african_codes=["GHA", "KEN", "NGA"])
    assert "Sub-Saharan Africa" not in result["Country Name"].values
    assert len(result) == 3


def test_filter_african_countries_excludes_mena(sample_wdi_df):
    """B7 FIX: MENA codes must not be in the African list."""
    # Add a MENA country to the test data
    mena_row = pd.DataFrame({
        "Country Name": ["Saudi Arabia"],
        "Country Code": ["SAU"],
        "Indicator Name": ["GDP per capita growth"],
        "Indicator Code": ["NY.GDP.PCAP.KD.ZG"],
        "2018": [1.0],
        "2019": [2.0],
    })
    df_with_mena = pd.concat([sample_wdi_df, mena_row], ignore_index=True)
    result = filter_african_countries(df_with_mena, african_codes=["GHA", "KEN", "NGA"])
    assert "SAU" not in result["Country Code"].values
    assert len(result) == 3


def test_filter_indicators_selects_correct_codes(sample_wdi_df):
    """Only selected indicator codes should remain."""
    codes = ["NY.GDP.PCAP.KD.ZG"]
    result = filter_indicators(sample_wdi_df, codes)
    assert result["Indicator Code"].nunique() == 1


def test_reshape_wide_to_long(sample_wdi_df):
    """Wide year columns should become a 'year' column."""
    result = reshape_wide_to_long(sample_wdi_df, year_columns=["2018", "2019"])
    assert "year" in result.columns
    assert np.issubdtype(result["year"].dtype, np.integer) or np.issubdtype(result["year"].dtype, np.floating)
    # 4 countries x 1 indicator x 2 years = 8
    assert len(result) == 8


def test_clean_numeric_handles_blank_strings():
    """Blank strings and placeholders should become NaN."""
    df = pd.DataFrame({"value": ["1.5", "", "..", "N/A", "3.2"]})
    result = clean_numeric(df, ["value"])
    assert pd.isna(result["value"].iloc[1])  # blank
    assert pd.isna(result["value"].iloc[2])  # ..
    assert pd.isna(result["value"].iloc[3])  # N/A
    assert result["value"].iloc[0] == pytest.approx(1.5)
    assert result["value"].iloc[4] == pytest.approx(3.2)


def test_log_dataset_info_logs(caplog):
    """log_dataset_info should log dataset dimensions."""
    import logging
    df = pd.DataFrame({"iso3": ["GHA", "KEN"], "year": [2018, 2019], "val": [1.0, 2.0]})
    with caplog.at_level(logging.INFO):
        log_dataset_info(df, "Test")
    assert "Shape" in caplog.text

def test_check_duplicates_flags_conflicting_values(caplog):
    """M6: conflicting duplicate keys must be surfaced at WARNING level."""
    import logging
    df = pd.DataFrame({
        "iso3": ["GHA", "GHA", "KEN"],
        "year": [2018, 2018, 2018],
        "indicator_code": ["X", "X", "X"],
        "value": [1.0, 2.0, 3.0],          # conflicting
    })
    with caplog.at_level(logging.WARNING):
        dups = check_duplicates(df, ["iso3", "year", "indicator_code"])
    assert len(dups) == 2
    assert "conflicting" in caplog.text.lower()


def test_check_duplicates_clean_panel_returns_empty():
    """No duplicate keys -> empty result, no warning."""
    df = pd.DataFrame({
        "iso3": ["GHA", "KEN"], "year": [2018, 2018],
        "indicator_code": ["X", "X"], "value": [1.0, 2.0],
    })
    assert check_duplicates(df, ["iso3", "year", "indicator_code"]).empty


def test_filter_african_countries_ignores_aggregates_without_explicit_blocklist():
    """M12: aggregates cannot appear because the config list only holds
    sovereign codes; filtering must not re-introduce them."""
    df = pd.DataFrame({
        "Country Name": ["Ghana", "Sub-Saharan Africa"],
        "Country Code": ["GHA", "SSF"],
        "Indicator Code": ["X", "X"],
    })
    result = filter_african_countries(df, african_codes=["GHA"])
    assert result["Country Code"].tolist() == ["GHA"]
