"""Tests for model training, pipelines, and serialization."""
import numpy as np
import pandas as pd
import pytest
import tempfile
from pathlib import Path
from src.train import (
    global_mean_baseline,
    persistence_baseline,
    build_ridge_pipeline,
    build_hgb_pipeline,
    train_and_evaluate,
    save_pipeline,
    load_pipeline,
    compute_metrics,
)


@pytest.fixture
def train_data():
    """Create sample training data."""
    np.random.seed(42)
    X = pd.DataFrame({
        "feat1": np.random.randn(100),
        "feat2": np.random.randn(100),
        "feat3": np.random.randn(100),
    })
    y = pd.Series(np.random.randn(100) * 2 + 5, name="target")
    return X, y


@pytest.fixture
def val_data():
    """Create sample validation data."""
    np.random.seed(99)
    X = pd.DataFrame({
        "feat1": np.random.randn(30),
        "feat2": np.random.randn(30),
        "feat3": np.random.randn(30),
    })
    y = pd.Series(np.random.randn(30) * 2 + 5, name="target")
    return X, y


def test_global_mean_baseline(train_data, val_data):
    """Global mean baseline should predict mean of training target."""
    X_train, y_train = train_data
    X_val, y_val = val_data
    pred = global_mean_baseline(y_train, n_predictions=len(y_val))
    assert len(pred) == len(y_val)
    assert np.allclose(pred, y_train.mean())


def test_persistence_baseline_uses_current_year_growth():
    """B2 FIX: Persistence baseline should predict current year's growth,
    not copy arbitrary historical values."""
    # Simulate: current year growth for test countries
    current_growth = pd.Series([3.2, 5.1, -1.4, 2.8, 4.0])
    result = persistence_baseline(current_growth)
    # Each prediction should equal the corresponding current year value
    np.testing.assert_array_equal(result, current_growth.values)
    assert len(result) == 5


def test_build_ridge_pipeline():
    """Ridge pipeline should have imputer, scaler, and ridge steps."""
    pipeline = build_ridge_pipeline(alpha=1.0)
    assert len(pipeline.named_steps) >= 2
    # Verify it has the right step names
    assert "imputer" in pipeline.named_steps
    assert "model" in pipeline.named_steps


def test_build_hgb_pipeline():
    """HGB pipeline should have imputer and gradient boosting steps."""
    pipeline = build_hgb_pipeline(max_iter=100, random_state=42)
    assert len(pipeline.named_steps) >= 2


def test_ridge_pipeline_handles_nan(train_data):
    """Pipeline should handle NaN values via imputer."""
    X_train, y_train = train_data
    X_train_nan = X_train.copy()
    X_train_nan.iloc[0, 0] = np.nan  # Add a NaN
    pipeline = build_ridge_pipeline()
    pipeline.fit(X_train_nan, y_train)
    pred = pipeline.predict(X_train_nan[:5])
    assert len(pred) == 5
    assert not np.any(np.isnan(pred))


def test_train_and_evaluate_returns_metrics(train_data, val_data):
    """Training should return a dictionary of evaluation metrics."""
    X_train, y_train = train_data
    X_val, y_val = val_data
    pipeline = build_ridge_pipeline()
    metrics = train_and_evaluate(pipeline, X_train, y_train, X_val, y_val)
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics
    assert "directional_accuracy" in metrics
    assert metrics["mae"] >= 0


def test_save_and_load_pipeline(train_data):
    """Pipeline should survive a save/load roundtrip."""
    X_train, y_train = train_data
    pipeline = build_ridge_pipeline()
    pipeline.fit(X_train, y_train)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "model.joblib"
        save_pipeline(pipeline, path)
        loaded = load_pipeline(path)
        pred_original = pipeline.predict(X_train[:5])
        pred_loaded = loaded.predict(X_train[:5])
        np.testing.assert_array_almost_equal(pred_original, pred_loaded)


def test_compute_metrics():
    """Metrics should be computed correctly."""
    actual = np.array([1.0, 2.0, 3.0, 4.0])
    predicted = np.array([1.1, 2.2, 2.8, 3.9])
    metrics = compute_metrics(actual, predicted)
    assert metrics["mae"] == pytest.approx(0.15, abs=0.01)
    assert 0 <= metrics["directional_accuracy"] <= 1

def test_get_transformed_feature_names_reflects_column_permutation():
    """ColumnTransformer moves the log column to position 0; names must follow."""
    from src.train import get_transformed_feature_names
    feats = ["A", "B", "C", "D"]
    pipe = build_ridge_pipeline(alpha=1.0, log_transform_features=["C"],
                                all_feature_names=feats)
    X = pd.DataFrame(np.random.rand(20, 4), columns=feats)
    pipe.fit(X, pd.Series(np.random.rand(20)))
    names = get_transformed_feature_names(pipe, feats)
    assert names == ["C_log1p", "A", "B", "D"], names
    assert len(names) == len(pipe.named_steps["model"].coef_)


def test_get_transformed_feature_names_identity_without_log_step():
    """Without the log_transform step the order is unchanged."""
    from src.train import get_transformed_feature_names
    feats = ["A", "B"]
    pipe = build_ridge_pipeline(alpha=1.0)
    assert get_transformed_feature_names(pipe, feats) == feats


def test_directional_metrics_expose_majority_class_degeneracy():
    """A constant positive predictor must score 0 skill, not high accuracy."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, -1.0])   # 80% positive
    y_pred = np.full(5, 2.0)                         # always positive
    m = compute_metrics(y_true, y_pred)
    assert m["directional_accuracy"] == pytest.approx(0.8)
    assert m["directional_majority_rate"] == pytest.approx(0.8)
    assert m["directional_skill"] == pytest.approx(0.0)
    assert m["balanced_directional_accuracy"] == pytest.approx(0.5)


def test_expanding_window_splits_are_chronological_and_disjoint():
    """Folds must never train on future years and must not overlap."""
    from src.train import expanding_window_splits
    years = pd.Series(list(range(2000, 2018)) * 3).sort_values().reset_index(drop=True)
    splits = expanding_window_splits(years, initial_train_end=2010, val_window=2)
    assert len(splits) >= 3
    for tr, va in splits:
        assert years.iloc[tr].max() < years.iloc[va].min()   # no future leakage
        assert not set(tr) & set(va)                          # disjoint


def test_search_hyperparameters_returns_ranked_grid():
    """Grid search results sorted by mean fold MAE, ascending."""
    from src.train import search_hyperparameters
    rng = np.random.RandomState(0)
    n = 180
    years = pd.Series(np.repeat(np.arange(2000, 2018), 10))
    X = pd.DataFrame({"a": rng.randn(n), "b": rng.randn(n)})
    y = pd.Series(X["a"] * 2 + rng.randn(n) * 0.5)
    res = search_hyperparameters(
        lambda **p: build_ridge_pipeline(**p),
        [{"alpha": 1.0}, {"alpha": 100.0}],
        X, y, years, initial_train_end=2010,
    )
    assert list(res.columns[:1]) == ["alpha"]
    assert {"mean_mae", "std_mae", "n_folds"} <= set(res.columns)
    assert res["mean_mae"].is_monotonic_increasing


def test_build_hgb_pipeline_has_explicit_early_stopping():
    """C5: early stopping must be a real bool with bounded iterations."""
    pipe = build_hgb_pipeline(max_iter=50, learning_rate=0.1, max_depth=2,
                              random_state=0)
    model = pipe.named_steps["model"]
    assert model.early_stopping is True
    assert model.validation_fraction == 0.15
    assert model.n_iter_no_change == 15
    X = pd.DataFrame(np.random.RandomState(0).randn(80, 3), columns=list("abc"))
    y = pd.Series(X["a"] + np.random.RandomState(1).randn(80) * 0.1)
    pipe.fit(X, y)
    assert pipe.named_steps["model"].n_iter_ <= 50


def test_baseline_gate_fails_when_candidate_worse():
    """C1 regression guard: a model worse than the constant must not pass."""
    from src.train import enforce_baseline_gate
    rep = enforce_baseline_gate({"mae": 3.54}, {"global_mean": {"mae": 1.90}})
    assert rep["passed"] is False
    assert "global_mean" in rep["failures"][0]


def test_baseline_gate_passes_when_candidate_better():
    from src.train import enforce_baseline_gate
    rep = enforce_baseline_gate(
        {"mae": 1.80},
        {"global_mean": {"mae": 1.90}, "persistence": {"mae": 2.23}},
    )
    assert rep["passed"] is True
    assert rep["per_baseline"]["global_mean"]["margin"] == pytest.approx(0.10)


def test_baseline_gate_reproduces_the_c1_regression():
    """Regression guard for the exact shipped failure."""
    from src.train import enforce_baseline_gate
    rep = enforce_baseline_gate(
        {"mae": 3.5397},
        {"global_mean": {"mae": 1.8958}, "persistence": {"mae": 2.2287}},
    )
    assert rep["passed"] is False
    assert len(rep["failures"]) == 2


def test_baseline_gate_requires_baselines():
    from src.train import enforce_baseline_gate
    with pytest.raises(ValueError):
        enforce_baseline_gate({"mae": 1.0}, {})


def test_write_model_metadata_includes_provenance(tmp_path):
    """M7: metadata records creation time, versions, and split target years."""
    import json
    from src.train import write_model_metadata
    p = tmp_path / "meta.json"
    write_model_metadata(
        path=p, feature_names=["a"], target_code="X", train_end=2017,
        val_end=2020, metrics={}, model_type="Ridge", random_state=42,
    )
    meta = json.loads(p.read_text())
    for key in ("created_utc", "library_versions", "split_target_years"):
        assert key in meta, key


def test_write_model_metadata_stores_gate_and_significance(tmp_path):
    """Gate/significance blocks round-trip verbatim when provided."""
    import json
    from src.train import write_model_metadata
    p = tmp_path / "meta.json"
    gate = {"metric": "mae", "passed": False, "failures": ["global_mean"],
            "per_baseline": {}}
    sig = {"paired_mae_improvement_vs_global_mean": 0.075,
           "ci_lower": -0.047, "ci_upper": 0.193, "significant_at_95": False,
           "n_bootstrap": 5000}
    write_model_metadata(
        path=p, feature_names=["a"], target_code="X", train_end=2017,
        val_end=2020, metrics={}, model_type="HGB", random_state=42,
        gate=gate, significance=sig, refit_strategy="train_only",
    )
    meta = json.loads(p.read_text())
    assert meta["gate"] == gate
    assert meta["significance"] == sig
    assert meta["refit_strategy"] == "train_only"
