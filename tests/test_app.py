"""Tests for app.py pure helpers (H5 remediation).

Covers the serving-side contracts the review flagged as untested:
scenario-input column ordering, NaN preservation for the imputer, training-
window guardrails (H3), extrapolation warnings, display fallbacks, and an
end-to-end prediction through the committed artifacts.

The Streamlit runtime is not started; module-level code executes in Streamlit
"bare mode". Guarded by importorskip so the suite passes on a test-only env.
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("streamlit")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

app = importlib.import_module("app")


# ----------------------------------------------------------------------------
# Scenario input contract
# ----------------------------------------------------------------------------

def test_prepare_scenario_input_column_order_matches_contract():
    """Model contract is positional+named: output order must equal feature_names."""
    feats = ["b_code", "a_code", "c_code"]
    baseline = {"a_code": 1.0, "b_code": 2.0, "c_code": 3.0}
    changes = {"a_code": 9.9}
    out = app.prepare_scenario_input(baseline, changes, feats)
    assert list(out.columns) == feats
    assert len(out) == 1
    assert out["a_code"].iloc[0] == 9.9            # scenario applied
    assert out["b_code"].iloc[0] == 2.0            # baseline preserved


def test_prepare_scenario_input_preserves_nan_for_imputer():
    """Missing features must arrive as NaN (in-pipeline median imputation)."""
    feats = ["x", "y"]
    out = app.prepare_scenario_input({"x": 1.0}, {}, feats)
    assert np.isnan(out["y"].iloc[0])


def test_prepare_scenario_input_ignores_unknown_scenario_keys():
    feats = ["x"]
    out = app.prepare_scenario_input({"x": 1.0}, {"not_a_feature": 5.0}, feats)
    assert list(out.columns) == ["x"]


# ----------------------------------------------------------------------------
# Guardrails: training-window calibration (H3)
# ----------------------------------------------------------------------------

@pytest.fixture()
def panel_with_regime_split():
    """Inflation moderate in train (max 40), extreme in test years (92)."""
    years = list(range(2000, 2025))
    rows = []
    for y in years:
        val = 30.0 if y <= 2017 else (5.0 if y <= 2020 else 92.0)
        rows.append({"iso3": "GHA", "country_name": "Ghana", "year": y,
                     "FP.CPI.TOTL.ZG": val})
    return pd.DataFrame(rows)


def test_get_training_data_filters_years(panel_with_regime_split):
    train = app.get_training_data(panel_with_regime_split, 2017)
    assert train["year"].max() <= 2017
    assert len(train) == 18


def test_get_feature_percentiles_uses_training_window_only(panel_with_regime_split):
    """H3: full-panel P99 for inflation is 92.05-style; training-only is 30."""
    train = app.get_training_data(panel_with_regime_split, 2017)
    p1_full, p99_full = app.get_feature_percentiles(panel_with_regime_split,
                                                     "FP.CPI.TOTL.ZG")
    p1_tr, p99_tr = app.get_feature_percentiles(train, "FP.CPI.TOTL.ZG")
    assert p99_full == pytest.approx(92.0)     # the WRONG (permissive) band
    assert p99_tr == pytest.approx(30.0)       # the guardrail we must use


def test_check_extrapolation_warning_fires_outside_training_band(panel_with_regime_split):
    train = app.get_training_data(panel_with_regime_split, 2017)
    warns = app.check_extrapolation_warning({"FP.CPI.TOTL.ZG": 80.0}, train)
    assert len(warns) == 1 and "training" in warns[0]
    ok = app.check_extrapolation_warning({"FP.CPI.TOTL.ZG": 30.0}, train)
    assert ok == []


def test_get_feature_range_training_window(panel_with_regime_split):
    train = app.get_training_data(panel_with_regime_split, 2017)
    lo, hi = app.get_feature_range(train, "FP.CPI.TOTL.ZG")
    assert (lo, hi) == (30.0, 30.0)
    lo_full, hi_full = app.get_feature_range(panel_with_regime_split, "FP.CPI.TOTL.ZG")
    assert hi_full == 92.0 and hi < hi_full   # training band strictly tighter


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------

def test_get_scenario_features_returns_three_to_five():
    meta = {"feature_names": ["EG.ELC.ACCS.ZS", "IT.NET.USER.ZS",
                              "NE.GDI.TOTL.ZS", "NE.TRD.GNFS.ZS",
                              "FP.CPI.TOTL.ZG", "SP.DYN.LE00.IN",
                              "NY.GDP.PCAP.CD"]}
    out = app.get_scenario_features(meta, [])
    assert 3 <= len(out) <= 5
    assert set(out) <= set(meta["feature_names"])


def test_get_scenario_features_pads_short_lists():
    """Fewer than 3 priority features: fall back to other available columns."""
    meta = {"feature_names": ["EG.ELC.ACCS.ZS", "XX.OTHER.A", "YY.OTHER.B",
                              "ZZ.OTHER.C"]}
    out = app.get_scenario_features(meta, [])
    assert len(out) >= 3


def test_get_feature_display_name_falls_back_to_code():
    feats = [{"code": "A.B.C", "name": "Alpha"}]
    assert app.get_feature_display_name("A.B.C", feats) == "Alpha"
    assert app.get_feature_display_name("Q.R.S", feats) == "Q.R.S"


def test_safe_get_country_data_returns_none_when_missing():
    df = pd.DataFrame({"iso3": ["GHA"], "year": [2019], "v": [1.0]})
    assert app.safe_get_country_data(df, "GHA", 2019) is not None
    assert app.safe_get_country_data(df, "KEN", 2019) is None
    assert app.safe_get_country_data(df, "GHA", 1999) is None


# ----------------------------------------------------------------------------
# End-to-end through committed artifacts (no retraining)
# ----------------------------------------------------------------------------

def test_end_to_end_prediction_from_committed_artifacts():
    """Deployed pipeline + metadata contract serve a finite prediction."""
    import joblib
    import json
    meta = json.loads((REPO_ROOT / "models/model_metadata.json").read_text())
    pipe = joblib.load(REPO_ROOT / "models/growth_model.joblib")
    panel = pd.read_parquet(REPO_ROOT / "data/processed/model_data.parquet")
    feats = meta["feature_names"]
    row = panel[(panel["iso3"] == "GHA") & (panel["year"] == 2019)].iloc[0]
    baseline = {f: float(row[f]) if pd.notna(row.get(f, np.nan)) else np.nan
                for f in feats}
    X = app.prepare_scenario_input(baseline, {}, feats)
    pred = float(pipe.predict(X)[0])
    assert np.isfinite(pred)
    # and the scenario probe still respects the contract
    X2 = app.prepare_scenario_input(baseline, {feats[0]: 500.0}, feats)
    assert np.isfinite(float(pipe.predict(X2)[0]))
    assert list(X2.columns) == feats


def test_load_feature_importance_returns_ci_frame():
    """App-facing parquet carries significance flags (H1 display contract)."""
    df = pd.read_parquet(REPO_ROOT / "models/feature_importance.parquet")
    assert {"is_significant", "ci_lower", "ci_upper"} <= set(df.columns)
    # the app must not choke on the new schema
    out = app.load_feature_importance.__wrapped__()
    assert isinstance(out, pd.DataFrame) and len(out) > 0
