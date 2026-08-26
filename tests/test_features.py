"""Tests for feature engineering and target creation."""
import pandas as pd
import numpy as np
import pytest
from src.features import (
    create_target,
    select_features_by_coverage,
    build_feature_matrix,
    create_temporal_split,
)


@pytest.fixture
def sample_panel():
    """Create a sample country-year panel for testing."""
    np.random.seed(42)
    countries = ["GHA", "KEN", "NGA"]
    years = list(range(2000, 2022))
    rows = []
    for c in countries:
        for y in years:
            rows.append({
                "iso3": c,
                "country_name": c,
                "year": y,
                "NY.GDP.PCAP.KD.ZG": np.random.uniform(-2, 8),
                "EG.ELC.ACCS.ZS": np.random.uniform(40, 100),
                "FP.CPI.TOTL.ZG": np.random.uniform(0, 20),
            })
    return pd.DataFrame(rows)


def test_create_target_shifts_gdp_growth(sample_panel):
    """Target should be GDP growth shifted forward by 1 year."""
    result = create_target(sample_panel, "NY.GDP.PCAP.KD.ZG")
    assert "target_next_year" in result.columns
    last_year_mask = result["year"] == result["year"].max()
    assert result.loc[last_year_mask, "target_next_year"].isna().all()
    non_last_mask = result["year"] < result["year"].max()
    assert result.loc[non_last_mask, "target_next_year"].notna().all()


def test_select_features_by_coverage_drops_sparse(sample_panel):
    """Features with low coverage should be dropped."""
    # Create a feature with exactly 50% coverage (below 60% threshold)
    n = len(sample_panel)
    sparse_col = np.array([1.0 if i % 2 == 0 else np.nan for i in range(n)])
    sample_panel["SPARSE_FEATURE"] = sparse_col
    result = select_features_by_coverage(
        sample_panel,
        feature_cols=["NY.GDP.PCAP.KD.ZG", "EG.ELC.ACCS.ZS", "SPARSE_FEATURE"],
        min_coverage=0.6,
    )
    assert "SPARSE_FEATURE" not in result.columns
    assert "NY.GDP.PCAP.KD.ZG" in result.columns


def test_select_features_by_coverage_uses_train_only(sample_panel):
    """MINOR FIX: Coverage should be computed on training rows only."""
    # Make all values in train have coverage but test has NaNs
    sample_panel.loc[sample_panel["year"] > 2015, "EG.ELC.ACCS.ZS"] = np.nan
    # If computed on full panel, EG.ELC.ACCS.ZS would be dropped
    # If computed on train only (<=2015), it should be kept
    result = select_features_by_coverage(
        sample_panel,
        feature_cols=["NY.GDP.PCAP.KD.ZG", "EG.ELC.ACCS.ZS"],
        min_coverage=0.6,
        train_mask=sample_panel["year"] <= 2015,
    )
    assert "EG.ELC.ACCS.ZS" in result.columns


def test_create_temporal_split_respects_years(sample_panel):
    """Temporal split should separate by year boundaries."""
    sample_panel = create_target(sample_panel, "NY.GDP.PCAP.KD.ZG")
    train, val, test = create_temporal_split(
        sample_panel, train_end=2015, val_end=2018
    )
    assert train["year"].max() <= 2015
    assert val["year"].min() >= 2016
    assert val["year"].max() <= 2018
    assert test["year"].min() >= 2019


def test_build_feature_matrix_drops_nan_target(sample_panel):
    """Rows with NaN target should be dropped."""
    sample_panel = create_target(sample_panel, "NY.GDP.PCAP.KD.ZG")
    X, y = build_feature_matrix(
        sample_panel,
        ["NY.GDP.PCAP.KD.ZG", "EG.ELC.ACCS.ZS"],
    )
    assert y.notna().all()
    assert len(X) == len(y)