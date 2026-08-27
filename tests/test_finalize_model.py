"""Tests for scripts/finalize_model.py (H5 remediation).

These are the regression guards the review said were missing: C1 (baseline
gate must block artifact writing), C2 (selection must not depend on the test
split), C5 (early stopping / artifact contracts), and metadata consistency.
All tests run main() against synthetic panels in tmp_path; the committed
panel is never modified.
"""
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from scripts.finalize_model import main

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG = REPO_ROOT / "config/indicators.yaml"

YEARS = np.arange(2000, 2025)
COUNTRIES = [f"ST{i:02d}" for i in range(20)]


def _make_learnable_panel(path: Path, seed: int = 7) -> Path:
    """20 countries x 25 years panel where next-year growth IS predictable.

    Construction: AR(1) latent a[t]; growth g[t] = 2*a[t] + noise. Because
    a[t+1] is predictable from a[t], a[t] carries real t+1 information, so a
    tuned model beats the global-mean and persistence baselines on validation
    and the gate should pass.
    """
    rng = np.random.RandomState(seed)
    n = len(COUNTRIES) * len(YEARS)
    rows = []
    for c in COUNTRIES:
        a = np.zeros(len(YEARS))
        for t in range(1, len(YEARS)):
            a[t] = 0.8 * a[t - 1] + rng.randn() * 0.6
        g = 2 * a + rng.randn(len(YEARS)) * 0.4
        noise = rng.randn(len(YEARS))
        sparse = np.where(rng.rand(len(YEARS)) < 0.8, np.nan, 5.0)  # ~20% coverage
        for t, y in enumerate(YEARS):
            rows.append({
                "iso3": c, "country_name": f"State {c}", "year": int(y),
                "NY.GDP.PCAP.KD.ZG": g[t],
                "EG.ELC.ACCS.ZS": a[t],
                "FP.CPI.TOTL.ZG": noise[t],
                # below-threshold coverage: must be dropped by the filter
                "SL.UEM.TOTL.ZS": sparse[t],
            })
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)
    return path


def _make_unlearnable_panel(path: Path, seed: int = 11) -> Path:
    """Panel with NO exploitable signal plus a train->val regime shift.

    Growth is pure iid noise in every year (uncorrelated with features and
    across time), and the validation target mean is shifted +3 relative to
    train. Any model fitted on train predicts the train mean and therefore
    CANNOT beat the global-mean baseline on validation: the gate must fail.
    """
    rng = np.random.RandomState(seed)
    rows = []
    for c in COUNTRIES:
        g = rng.randn(len(YEARS)).copy()
        g[(YEARS >= 2018) & (YEARS <= 2020)] += 3.0   # regime shift in val
        for t, y in enumerate(YEARS):
            rows.append({
                "iso3": c, "country_name": f"State {c}", "year": int(y),
                "NY.GDP.PCAP.KD.ZG": g[t],
                "EG.ELC.ACCS.ZS": rng.randn(),
                "FP.CPI.TOTL.ZG": rng.randn(),
            })
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)
    return path


@pytest.fixture(scope="module")
def learnable_panel(tmp_path_factory):
    p = tmp_path_factory.mktemp("panel") / "learnable.parquet"
    return _make_learnable_panel(p)


@pytest.fixture(scope="module")
def noise_panel(tmp_path_factory):
    p = tmp_path_factory.mktemp("panel") / "noise.parquet"
    return _make_unlearnable_panel(p)


def _run(panel: Path, out: Path, allow_fail: bool = False) -> dict:
    return main(config_path=CONFIG, panel_path=panel, output_dir=out,
                data_dir=out, allow_baseline_failure=allow_fail)


def test_finalize_writes_all_expected_artifacts(learnable_panel, tmp_path):
    out = tmp_path / "models"
    _run(learnable_panel, out)
    for name in ["growth_model.joblib", "model_metadata.json",
                 "test_predictions.parquet", "feature_importance.parquet",
                 "ridge_coefficients.parquet", "cv_results_ridge.csv",
                 "cv_results_hgb.csv", "country_metadata.csv"]:
        assert (out / name).exists(), name


def test_finalize_passes_gate_on_learnable_panel(learnable_panel, tmp_path):
    out = tmp_path / "models"
    meta = _run(learnable_panel, out)
    assert meta["gate"]["passed"] is True
    assert meta["metrics"]["winner_val"]["mae"] < \
        meta["metrics"]["val_baselines"]["global_mean"]["mae"]


def test_finalize_exits_nonzero_when_gate_fails(noise_panel, tmp_path):
    """Pure-noise target must fail the gate and refuse to write artifacts."""
    out = tmp_path / "models"
    with pytest.raises(SystemExit) as e:
        _run(noise_panel, out)
    assert e.value.code == 2
    assert not (out / "growth_model.joblib").exists()
    assert not (out / "model_metadata.json").exists()
    assert not (out / "test_predictions.parquet").exists()


def test_finalize_allow_flag_writes_but_records_failure(noise_panel, tmp_path):
    out = tmp_path / "models"
    meta = _run(noise_panel, out, allow_fail=True)
    assert meta["gate"]["passed"] is False
    assert meta["gate"]["failures"]
    assert (out / "growth_model.joblib").exists()


def test_metadata_feature_contract_matches_pipeline(learnable_panel, tmp_path):
    """Deployed pipeline must accept exactly metadata['feature_names']."""
    out = tmp_path / "models"
    meta = _run(learnable_panel, out)
    pipe = joblib.load(out / "growth_model.joblib")
    feats = meta["feature_names"]
    assert "SL.UEM.TOTL.ZS" not in feats          # low-coverage dropped
    X = pd.DataFrame({f: np.full(5, 1.0) for f in feats})
    pred = pipe.predict(X)
    assert pred.shape == (5,) and np.isfinite(pred).all()


def test_test_predictions_rows_match_test_split(learnable_panel, tmp_path):
    out = tmp_path / "models"
    meta = _run(learnable_panel, out)
    preds = pd.read_parquet(out / "test_predictions.parquet")
    assert len(preds) == meta["split_sizes"]["test"]
    assert {"iso3", "year", "actual", "predicted"} <= set(preds.columns)
    assert preds["year"].min() > meta["val_end"]


def test_importance_and_direction_blocks_exist(learnable_panel, tmp_path):
    out = tmp_path / "models"
    _run(learnable_panel, out)
    imp = pd.read_parquet(out / "feature_importance.parquet")
    assert {"ci_lower", "ci_upper", "is_significant"} <= set(imp.columns)
    rc = pd.read_parquet(out / "ridge_coefficients.parquet")
    assert len(rc) == len(imp)                    # one coef per feature


def test_selection_uses_validation_not_test(learnable_panel, tmp_path):
    """Adversarial C2 guard: corrupt test targets; selection must be identical.

    We negate and inflate every test-window target value. If any selection,
    refit, or interpretation step consulted test, the deployed model_type,
    validation metrics, gate report, or CV tables would change. They must not.
    """
    panel = pd.read_parquet(learnable_panel)
    corrupted = panel.copy()
    # Corrupt only target-space years reachable from test feature rows
    # (2022-2024). Years 2020/2021 growth are val targets -> left untouched.
    test_mask = corrupted["year"] >= 2022
    corrupted.loc[test_mask, "NY.GDP.PCAP.KD.ZG"] = \
        -50.0 - corrupted.loc[test_mask, "NY.GDP.PCAP.KD.ZG"].abs()
    cpath = tmp_path / "corrupted.parquet"
    corrupted.to_parquet(cpath, index=False)

    out_a, out_b = tmp_path / "a", tmp_path / "b"
    meta_a = _run(learnable_panel, out_a)
    meta_b = _run(cpath, out_b)

    assert meta_a["model_type"] == meta_b["model_type"]
    for key in ("ridge_val", "hgb_val", "winner_val"):
        assert meta_a["metrics"][key] == meta_b["metrics"][key], key
    assert meta_a["gate"] == meta_b["gate"]
    # CV result tables must be byte-identical
    for f in ["cv_results_ridge.csv", "cv_results_hgb.csv"]:
        assert (out_a / f).read_bytes() == (out_b / f).read_bytes(), f
    # importance is computed on validation, so it too must be identical
    assert (out_a / "feature_importance.parquet").read_bytes() == \
        (out_b / "feature_importance.parquet").read_bytes()
    # ...while the test metrics legitimately differ (the set is what we broke)
    assert meta_a["metrics"]["winner_test"] != meta_b["metrics"]["winner_test"]


def test_determinism_of_finalize(learnable_panel, tmp_path):
    out_a, out_b = tmp_path / "a", tmp_path / "b"
    meta_a = _run(learnable_panel, out_a)
    meta_b = _run(learnable_panel, out_b)
    assert meta_a["metrics"] == meta_b["metrics"]
    assert meta_a["significance"] == meta_b["significance"]


def test_provenance_fields_populated(learnable_panel, tmp_path):
    out = tmp_path / "models"
    meta = _run(learnable_panel, out)
    for key in ["created_utc", "library_versions", "data_provenance",
                "split_sizes", "split_target_years", "refit_strategy",
                "feature_selection"]:
        assert key in meta, key
    assert meta["data_provenance"]["panel_sha256"]
    st = meta["split_target_years"]
    assert st["train"][1] - st["train"][0] == 17         # 2001..2018 inclusive
    assert st["val"] == [2019, 2021] and st["test"] == [2022, 2024]
