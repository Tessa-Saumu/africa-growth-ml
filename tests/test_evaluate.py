"""Tests for model evaluation and interpretation."""
import numpy as np
import pandas as pd
import pytest
from src.evaluate import (
    compute_metrics_by_group,
    compute_directional_accuracy,
    compute_bootstrap_ci,
    compute_permutation_importance,
    compute_worst_errors,
)


@pytest.fixture
def sample_predictions():
    """Create sample prediction data for evaluation."""
    np.random.seed(42)
    n = 100
    actual = np.random.randn(n) * 3 + 5
    predicted = actual + np.random.randn(n) * 0.5
    return pd.DataFrame({
        "actual": actual,
        "predicted": predicted,
        "iso3": np.random.choice(["GHA", "KEN", "NGA"], n),
        "year": np.random.choice(range(2018, 2024), n),
    })


def test_compute_metrics_by_group(sample_predictions):
    """Metrics should be computed per group."""
    result = compute_metrics_by_group(sample_predictions, group_col="iso3")
    assert "mae" in result.columns
    assert result["iso3"].nunique() == 3
    assert (result["mae"] >= 0).all()


def test_compute_directional_accuracy(sample_predictions):
    """Directional accuracy should be between 0 and 1."""
    result = compute_directional_accuracy(
        sample_predictions["actual"].values,
        sample_predictions["predicted"].values,
    )
    assert 0 <= result <= 1


def test_compute_bootstrap_ci_returns_bounds(sample_predictions):
    """Bootstrap CI should return lower and upper bounds."""
    lower, upper = compute_bootstrap_ci(
        sample_predictions["actual"].values,
        sample_predictions["predicted"].values,
        metric_fn=lambda a, p: np.mean(np.abs(a - p)),
        n_bootstrap=100,
    )
    assert lower < upper


def test_compute_permutation_importance_returns_series(sample_predictions):
    """Permutation importance should return a Series indexed by feature name."""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer

    X = pd.DataFrame({
        "feat_a": np.random.randn(100),
        "feat_b": np.random.randn(100),
    })
    y = sample_predictions["actual"]
    pipeline = Pipeline([("imputer", SimpleImputer()), ("model", Ridge())])
    pipeline.fit(X, y)

    importance = compute_permutation_importance(
        pipeline, X, y, feature_names=["feat_a", "feat_b"], n_repeats=5
    )
    assert isinstance(importance, pd.Series)
    assert "feat_a" in importance.index


def test_compute_worst_errors(sample_predictions):
    """Worst errors should return top-N rows by absolute error."""
    result = compute_worst_errors(sample_predictions, top_n=5)
    assert len(result) == 5
    assert "abs_error" in result.columns
    assert result["abs_error"].is_monotonic_decreasing

def test_compute_bootstrap_ci_logs_without_formatting_error(caplog):
    """M4: the CI log line must not raise a logging TypeError."""
    import logging
    a = np.random.RandomState(0).randn(30)
    p = a + 0.1
    with caplog.at_level(logging.INFO):
        compute_bootstrap_ci(a, p, lambda x, y: np.mean(np.abs(x - y)), n_bootstrap=25)
    assert any("Bootstrap CI" in r.getMessage() for r in caplog.records)
