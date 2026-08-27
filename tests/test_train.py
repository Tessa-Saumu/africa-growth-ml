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