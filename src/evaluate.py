"""Model evaluation, error analysis, and feature interpretation.

Computes metrics by group, bootstrap confidence intervals, permutation
importance, and worst-error analysis for model comparison.
"""
import numpy as np
import pandas as pd
import logging
from typing import Callable, Tuple, List
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance as sk_permutation_importance
from src.train import compute_metrics

logger = logging.getLogger(__name__)


def compute_metrics_by_group(
    df: pd.DataFrame,
    group_col: str = "iso3",
    actual_col: str = "actual",
    predicted_col: str = "predicted",
) -> pd.DataFrame:
    """Compute evaluation metrics grouped by a column (e.g., country or year).

    Args:
        df: DataFrame with actual and predicted columns.
        group_col: Column to group by.
        actual_col: Column name for actual values.
        predicted_col: Column name for predicted values.

    Returns:
        DataFrame with one row per group and columns for each metric.
    """
    results = []
    for group_name, group_df in df.groupby(group_col):
        metrics = compute_metrics(
            group_df[actual_col].values,
            group_df[predicted_col].values,
        )
        metrics[group_col] = group_name
        results.append(metrics)
    result_df = pd.DataFrame(results)
    logger.info("Computed metrics for %d groups in '%s'", len(result_df), group_col)
    return result_df


def compute_directional_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Compute the fraction of cases where growth direction is correctly predicted.

    Args:
        actual: Array of actual values.
        predicted: Array of predicted values.

    Returns:
        Fraction of correctly predicted directions (0.0 to 1.0).
    """
    correct = ((actual >= 0) == (predicted >= 0)).mean()
    logger.info("Directional accuracy: %.1f%%", correct * 100)
    return correct


def compute_bootstrap_ci(
    actual: np.ndarray,
    predicted: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> Tuple[float, float]:
    """Compute bootstrap confidence interval for a metric.

    Args:
        actual: Array of actual values.
        predicted: Array of predicted values.
        metric_fn: Function that takes (actual, predicted) and returns a scalar.
        n_bootstrap: Number of bootstrap resamples.
        confidence: Confidence level (e.g., 0.95 for 95% CI).
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (lower_bound, upper_bound).
    """
    rng = np.random.RandomState(random_state)
    n = len(actual)
    boot_scores = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        score = metric_fn(actual[idx], predicted[idx])
        boot_scores.append(score)
    boot_scores = np.array(boot_scores)
    alpha = (1 - confidence) / 2
    lower = np.percentile(boot_scores, alpha * 100)
    upper = np.percentile(boot_scores, (1 - alpha) * 100)
    logger.info(
        "Bootstrap CI (%.0f%%): [%.4f, %.4f] (n=%d)",
        confidence * 100, lower, upper, n_bootstrap
    )
    return lower, upper


def compute_permutation_importance(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: List[str],
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.Series:
    """Compute permutation importance for a fitted pipeline.

    Args:
        pipeline: Fitted sklearn Pipeline.
        X: Feature matrix.
        y: Target values.
        feature_names: List of feature names.
        n_repeats: Number of permutation repeats.
        random_state: Random seed.

    Returns:
        Series indexed by feature name with mean importance values.
    """
    result = sk_permutation_importance(
        pipeline, X, y,
        n_repeats=n_repeats,
        random_state=random_state,
    )
    importance = pd.Series(
        result.importances_mean, index=feature_names, name="importance"
    ).sort_values(ascending=False)
    logger.info("Permutation importance computed for %d features", len(importance))
    return importance


def compute_permutation_importance_with_ci(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: List[str],
    n_repeats: int = 30,
    random_state: int = 42,
) -> pd.DataFrame:
    """Permutation importance with dispersion and a significance flag.

    H1: importance magnitude carries no directional meaning, and values whose
    spread straddles zero are indistinguishable from noise. Callers must not
    render these as 'positive'/'negative' effects.

    Args:
        pipeline: Fitted sklearn Pipeline.
        X: Feature matrix to permute (use a *non-test* split for interpretation).
        y: Target aligned with X.
        feature_names: Ordered input feature names.
        n_repeats: Permutations per feature; drives CI width.
        random_state: Random seed for reproducibility.

    Returns:
        Columns: feature, importance_mean, importance_std, ci_lower, ci_upper,
        is_significant (ci_lower > 0), sorted by importance_mean descending.
    """
    res = sk_permutation_importance(
        pipeline, X, y, n_repeats=n_repeats, random_state=random_state)
    lo = np.percentile(res.importances, 2.5, axis=1)
    hi = np.percentile(res.importances, 97.5, axis=1)
    out = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": res.importances_mean,
        "importance_std": res.importances_std,
        "ci_lower": lo,
        "ci_upper": hi,
        "is_significant": lo > 0,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    logger.info("Permutation importance: %d/%d features significant at 95%%",
                int(out["is_significant"].sum()), len(out))
    return out


def compute_worst_errors(
    df: pd.DataFrame,
    top_n: int = 10,
    actual_col: str = "actual",
    predicted_col: str = "predicted",
) -> pd.DataFrame:
    """Identify the worst prediction errors.

    Args:
        df: DataFrame with actual and predicted columns.
        top_n: Number of worst errors to return.
        actual_col: Column name for actual values.
        predicted_col: Column name for predicted values.

    Returns:
        DataFrame with top_n rows sorted by absolute error descending.
    """
    df = df.copy()
    df["abs_error"] = np.abs(df[actual_col] - df[predicted_col])
    worst = df.nlargest(top_n, "abs_error")
    logger.info("Worst %d errors: MAE=%.4f, max=%.4f", top_n,
               worst["abs_error"].mean(), worst["abs_error"].max())
    return worst