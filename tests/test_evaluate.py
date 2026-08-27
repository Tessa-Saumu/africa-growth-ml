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


def test_permutation_importance_ci_flags_noise_features():
    """H1: noise features must NOT be flagged significant; real signal must be.

    NOTE: the remediation plan drafted this test evaluating importance on the
    training data itself. That is flaky across sklearn versions (tiny learned
    noise weights make permuting on-train data *worse*, i.e. spuriously
    'significant'), so we fit on a first half and measure on a held-out
    second half — the methodologically correct framing either way.
    """
    from src.evaluate import compute_permutation_importance_with_ci
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    rng = np.random.RandomState(0)
    n = 240
    n_fit = 160
    X = pd.DataFrame({"signal": rng.randn(n), "noise": rng.randn(n)})
    y = pd.Series(X["signal"] * 3 + rng.randn(n) * 0.2)
    pipe = Pipeline([("i", SimpleImputer()), ("m", Ridge())])
    pipe.fit(X.iloc[:n_fit], y.iloc[:n_fit])
    out = compute_permutation_importance_with_ci(
        pipe, X.iloc[n_fit:], y.iloc[n_fit:], ["signal", "noise"], n_repeats=20)
    assert out.iloc[0]["feature"] == "signal"
    assert bool(out.set_index("feature").loc["signal", "is_significant"]) is True
    assert bool(out.set_index("feature").loc["noise", "is_significant"]) is False


def test_permutation_importance_with_ci_columns():
    """Output carries the documented columns and sorted order."""
    from src.evaluate import compute_permutation_importance_with_ci
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    rng = np.random.RandomState(1)
    n = 120
    X = pd.DataFrame({"a": rng.randn(n), "b": rng.randn(n)})
    y = pd.Series(X["a"] * 2 + rng.randn(n) * 0.5)
    pipe = Pipeline([("i", SimpleImputer()), ("m", Ridge())])
    pipe.fit(X.iloc[:60], y.iloc[:60])
    out = compute_permutation_importance_with_ci(pipe, X.iloc[60:], y.iloc[60:],
                                                  ["a", "b"], n_repeats=10)
    assert list(out.columns) == ["feature", "importance_mean", "importance_std",
                                 "ci_lower", "ci_upper", "is_significant"]
    assert out["importance_mean"].is_monotonic_decreasing
