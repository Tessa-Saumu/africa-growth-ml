"""Model training, evaluation, and serialization.

Defines baseline predictors, sklearn pipelines for Ridge and HistGradientBoosting,
training loops with evaluation, and model artifact saving/loading.

B2 FIX: Persistence baseline predicts current year's growth (not last training values).
B3 FIX: Log transform is applied inside the sklearn pipeline via FunctionTransformer,
        so the app never sees _log columns. The pipeline consumes raw WDI column names.
"""
import numpy as np
import pandas as pd
import joblib
import json
import hashlib
import logging
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.features import clip_log1p  # B8 FIX: import from src/ for picklability

logger = logging.getLogger(__name__)


def global_mean_baseline(y_train: pd.Series, n_predictions: int) -> np.ndarray:
    """Predict the global training mean for all test observations.

    Args:
        y_train: Training target values.
        n_predictions: Number of predictions to generate.

    Returns:
        Array of constant predictions equal to training mean.
    """
    mean_val = y_train.mean()
    logger.info("Global mean baseline: predicting %.4f for %d observations",
               mean_val, n_predictions)
    return np.full(n_predictions, mean_val)


def persistence_baseline(current_year_growth: pd.Series) -> np.ndarray:
    """Predict next year's growth as this year's observed growth.

    B2 FIX: This is the correct persistence hypothesis:
    "next year's growth = this year's growth." For each test observation,
    the prediction is that row's current-year growth value.

    Args:
        current_year_growth: Current year's observed growth for each test row.

    Returns:
        Array of current year values (used directly as predictions).
    """
    logger.info("Persistence baseline: using current year values for %d predictions",
               len(current_year_growth))
    return current_year_growth.values


def build_ridge_pipeline(
    alpha: float = 1.0,
    log_transform_features: Optional[List[str]] = None,
    all_feature_names: Optional[List[str]] = None,
) -> Pipeline:
    """Build Ridge regression pipeline with imputation, optional log transform, scaling.

    B3+B8 FIX: Log transform is applied inside the pipeline via FunctionTransformer
    referencing clip_log1p from src/features.py (importable by module path for pickling).
    If log_transform_features is provided, those features get log1p via ColumnTransformer;
    others pass through.

    Args:
        alpha: Ridge regularization strength.
        log_transform_features: List of feature names to log-transform.
        all_feature_names: Full ordered list of feature names.

    Returns:
        sklearn Pipeline.
    """
    steps = [("imputer", SimpleImputer(strategy="median"))]

    if log_transform_features and all_feature_names:
        # Split features into log-transform and pass-through groups
        log_idx = [all_feature_names.index(f) for f in log_transform_features
                   if f in all_feature_names]
        pass_idx = [i for i in range(len(all_feature_names)) if i not in log_idx]

        preprocessor = ColumnTransformer(
            transformers=[
                ("log", FunctionTransformer(clip_log1p), log_idx),  # B8 FIX
                ("pass", "passthrough", pass_idx),
            ],
            remainder="drop",
        )
        steps.append(("log_transform", preprocessor))

    steps.append(("scaler", StandardScaler()))
    steps.append(("model", Ridge(alpha=alpha)))

    pipeline = Pipeline(steps)
    logger.info("Built Ridge pipeline (alpha=%.4f, log_features=%s)",
                alpha, log_transform_features)
    return pipeline


def get_transformed_feature_names(pipeline: Pipeline, input_features: List[str]) -> List[str]:
    """Return output feature names in the order the final estimator sees them.

    A ColumnTransformer concatenates transformer outputs in declaration order,
    so the log-transformed column is moved to position 0. Mapping coefficients
    with zip(input_features, coef_) is therefore incorrect whenever a
    'log_transform' step is present.

    Args:
        pipeline: Fitted or unfitted Pipeline, optionally containing a
            'log_transform' ColumnTransformer step.
        input_features: Feature names in the order passed to .fit().

    Returns:
        Output feature names aligned positionally with the final estimator's
        coef_ / feature_importances_ array.
    """
    if "log_transform" not in pipeline.named_steps:
        return list(input_features)
    ct = pipeline.named_steps["log_transform"]
    names: List[str] = []
    for name, _transformer, cols in ct.transformers_:
        if name == "remainder":
            continue
        for idx in cols:
            base = input_features[idx]
            names.append(f"{base}_log1p" if name == "log" else base)
    return names


def build_hgb_pipeline(
    max_iter: int = 200,
    learning_rate: float = 0.05,
    max_depth: int = 3,
    random_state: int = 42,
    l2_regularization: float = 1.0,
    early_stopping: bool = True,
    validation_fraction: float = 0.15,
    n_iter_no_change: int = 15,
) -> Pipeline:
    """Build HistGradientBoostingRegressor pipeline with imputation.

    C5 FIX: sklearn's `early_stopping="auto"` only activates above 10,000
    training samples, so on this panel (n≈900) it resolved to *disabled* and
    the model ran all 1000 boosting iterations unregularized-early — the
    mechanism behind the shipped overfit. Early stopping is therefore set
    explicitly here (boolean, with validation_fraction and n_iter_no_change).

    Args:
        max_iter: Maximum boosting iterations.
        learning_rate: Boosting learning rate.
        max_depth: Maximum tree depth.
        random_state: Random seed for reproducibility.
        l2_regularization: L2 penalty on leaf outputs.
        early_stopping: Enable explicit early stopping on an internal
            validation split. Must stay True for n < 10k samples.
        validation_fraction: Fraction of training data held out internally
            for early stopping.
        n_iter_no_change: Rounds without improvement before stopping.

    Returns:
        sklearn Pipeline.
    """
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingRegressor(
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            l2_regularization=l2_regularization,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
        )),
    ])
    logger.info(
        "Built HGB pipeline (max_iter=%d, lr=%.4f, depth=%d, l2=%.2f, es=%s)",
        max_iter, learning_rate, max_depth, l2_regularization, early_stopping
    )
    return pipeline


def expanding_window_splits(
    years: pd.Series,
    initial_train_end: int,
    val_window: int = 2,
    final_train_end: Optional[int] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate chronological expanding-window (train_idx, val_idx) pairs.

    Implements the spec section 9 protocol:
        train 2000-2010 -> validate 2011-2012
        train 2000-2012 -> validate 2013-2014
        ...
    All folds live strictly inside the training period; the held-out
    validation and test splits are never touched here.

    Args:
        years: Year value per row, index-aligned with the feature matrix.
        initial_train_end: Last year of the first training fold.
        val_window: Number of years in each validation fold.
        final_train_end: Last year available for folding (defaults to max year).

    Returns:
        List of (train_positions, val_positions) integer-position arrays.
    """
    y = years.reset_index(drop=True)
    last = int(final_train_end if final_train_end is not None else y.max())
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    cut = initial_train_end
    while cut + val_window <= last:
        tr = np.where(y <= cut)[0]
        va = np.where((y > cut) & (y <= cut + val_window))[0]
        if len(tr) and len(va):
            splits.append((tr, va))
        cut += val_window
    logger.info(
        "Expanding-window CV: %d folds (initial_train_end=%d, val_window=%d, last=%d)",
        len(splits), initial_train_end, val_window, last,
    )
    return splits


def search_hyperparameters(
    build_fn: Callable[..., Pipeline],
    param_grid: List[Dict[str, Any]],
    X: pd.DataFrame,
    y: pd.Series,
    years: pd.Series,
    initial_train_end: int,
    val_window: int = 2,
) -> pd.DataFrame:
    """Score a compact parameter grid with expanding-window CV.

    Selection uses mean fold MAE with std as a stability tiebreaker, per
    spec section 11 ("stability across temporal folds").

    Args:
        build_fn: Callable returning an unfitted Pipeline for given params.
        param_grid: Explicit list of parameter dicts (compact grid, not random).
        X: Training-period features only.
        y: Training-period target only.
        years: Year per row, index-aligned with X.
        initial_train_end: Last year of the first fold.
        val_window: Validation years per fold.

    Returns:
        One row per configuration with mean_mae, std_mae and fold count,
        sorted by mean_mae ascending.
    """
    splits = expanding_window_splits(years, initial_train_end, val_window)
    if not splits:
        raise ValueError("Expanding-window CV produced no folds; check year range.")

    rows = []
    for params in param_grid:
        fold_maes = []
        for tr_idx, va_idx in splits:
            model = build_fn(**params)
            model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            pred = model.predict(X.iloc[va_idx])
            fold_maes.append(mean_absolute_error(y.iloc[va_idx], pred))
        rows.append({
            **params,
            "mean_mae": float(np.mean(fold_maes)),
            "std_mae": float(np.std(fold_maes)),
            "n_folds": len(fold_maes),
        })
        logger.info("CV %s -> mean MAE %.4f (+/- %.4f)",
                    params, rows[-1]["mean_mae"], rows[-1]["std_mae"])
    return pd.DataFrame(rows).sort_values(["mean_mae", "std_mae"]).reset_index(drop=True)


def enforce_baseline_gate(
    candidate_metrics: Dict[str, float],
    baseline_metrics: Dict[str, Dict[str, float]],
    metric: str = "mae",
    lower_is_better: bool = True,
) -> Dict[str, Any]:
    """Fail the build unless the candidate beats every baseline on validation.

    C1: the previous pipeline shipped a model 86 percent worse than the global
    mean because baselines were only computed on test, after selection. This
    gate runs on validation, before any artifact is written.

    Args:
        candidate_metrics: Validation metrics for the selected model.
        baseline_metrics: Mapping of baseline name -> validation metrics.
        metric: Metric key to gate on.
        lower_is_better: True for error metrics.

    Returns:
        Report dict with 'passed' plus per-baseline margins.

    Raises:
        ValueError: If a baseline is missing the gating metric.
    """
    if not baseline_metrics:
        raise ValueError("Baseline gate requires at least one baseline.")

    cand = candidate_metrics[metric]
    results, failures = {}, []
    for name, bm in baseline_metrics.items():
        if metric not in bm:
            raise ValueError(f"Baseline '{name}' missing metric '{metric}'.")
        base = bm[metric]
        beats = cand < base if lower_is_better else cand > base
        margin = (base - cand) if lower_is_better else (cand - base)
        results[name] = {
            "baseline_value": float(base),
            "candidate_value": float(cand),
            "margin": float(margin),
            "relative_margin_pct": float(margin / base * 100) if base else float("nan"),
            "passed": bool(beats),
        }
        if not beats:
            failures.append(f"{name} ({metric}={base:.4f} vs candidate {cand:.4f})")
        logger.info("Baseline gate [%s]: candidate %.4f vs %.4f -> %s",
                    name, cand, base, "PASS" if beats else "FAIL")

    report = {"metric": metric, "passed": not failures,
              "failures": failures, "per_baseline": results}
    if failures:
        logger.error("BASELINE GATE FAILED against: %s", "; ".join(failures))
    else:
        logger.info("Baseline gate PASSED against all %d baselines.", len(results))
    return report


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute regression evaluation metrics.

    Note: `directional_accuracy` is NOT a skill measure — for a target
    distribution that is mostly one sign, any constant-sign predictor scores
    the majority-class rate by construction. It must always be quoted next to
    `directional_majority_rate`, or better, via `directional_skill` /
    `balanced_directional_accuracy`.

    Args:
        y_true: Actual target values.
        y_pred: Predicted values.

    Returns:
        Dictionary with mae, rmse, r2, directional_accuracy,
        directional_majority_rate, directional_skill, and
        balanced_directional_accuracy.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    direction_correct = ((y_true >= 0) == (y_pred >= 0)).mean()

    # H4: raw directional accuracy equals the majority-class rate for any
    # constant-sign predictor. Report skill-aware companions alongside it.
    actual_pos = y_true >= 0
    pred_pos = y_pred >= 0
    majority_rate = max(actual_pos.mean(), 1.0 - actual_pos.mean())

    tpr = (pred_pos[actual_pos]).mean() if actual_pos.any() else np.nan
    tnr = (~pred_pos[~actual_pos]).mean() if (~actual_pos).any() else np.nan
    balanced = np.nanmean([tpr, tnr])

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "directional_accuracy": direction_correct,
        "directional_majority_rate": float(majority_rate),
        "directional_skill": float(direction_correct - majority_rate),
        "balanced_directional_accuracy": float(balanced),
    }


def train_and_evaluate(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Dict[str, float]:
    """Train a pipeline and evaluate on validation data.

    Args:
        pipeline: sklearn Pipeline to train.
        X_train: Training features.
        y_train: Training target.
        X_val: Validation features.
        y_val: Validation target.

    Returns:
        Dictionary of evaluation metrics on validation set.
    """
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)
    metrics = compute_metrics(y_val.values, y_pred)
    logger.info(
        "Model evaluation - MAE: %.4f, RMSE: %.4f, R2: %.4f, Direction: %.1f%%",
        metrics["mae"], metrics["rmse"], metrics["r2"],
        metrics["directional_accuracy"] * 100
    )
    return metrics


def save_pipeline(pipeline: Pipeline, path: Path) -> None:
    """Save a fitted pipeline to disk using joblib.

    Args:
        pipeline: Fitted sklearn Pipeline.
        path: File path for the saved artifact.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    logger.info("Pipeline saved to %s (%.1f KB)", path, path.stat().st_size / 1024)


def load_pipeline(path: Path) -> Pipeline:
    """Load a fitted pipeline from disk.

    Args:
        path: File path of the saved artifact.

    Returns:
        Fitted sklearn Pipeline.
    """
    pipeline = joblib.load(path)
    logger.info("Pipeline loaded from %s", path)
    return pipeline


def _sha256(path: Path) -> Optional[str]:
    """Return the SHA-256 hex digest of a file, or None if unreadable.

    Args:
        path: File to hash.

    Returns:
        64-char hex digest, or None when the file cannot be read.
    """
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError as e:  # missing/unreadable file must not crash metadata write
        logger.warning("Could not hash %s: %s", path, e)
        return None


def _git_commit_or_none() -> Optional[str]:
    """Return the current git commit hash, or None if git is unavailable.

    Returns:
        Commit hash string, or None on any failure (never raises).
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        return out.stdout.strip() or None
    except Exception as e:  # noqa: BLE001 - provenance must never break the build
        logger.warning("Could not read git commit: %s", e)
        return None


def write_model_metadata(
    path: Path,
    feature_names: List[str],
    target_code: str,
    train_end: int,
    val_end: int,
    metrics: Dict[str, float],
    model_type: str,
    random_state: int,
    log_transform_features: Optional[List[str]] = None,
    data_provenance: Optional[Dict[str, Any]] = None,
    split_sizes: Optional[Dict[str, int]] = None,
    split_target_years: Optional[Dict[str, List[int]]] = None,
    refit_strategy: Optional[str] = None,
    gate: Optional[Dict[str, Any]] = None,
    significance: Optional[Dict[str, Any]] = None,
    sensitivity: Optional[Dict[str, Any]] = None,
    extras: Optional[Dict[str, Any]] = None,
) -> None:
    """Write model metadata JSON with feature contract, metrics, and provenance.

    B5 FIX: This file is the single source of truth for the model's feature
    contract. The app reads feature names from here, not from config.

    M7 FIX: metadata now carries creation timestamp, git commit, library
    versions, a data provenance block (panel path + sha256 + coverage stats),
    split sizes, explicit *target*-year windows per split (the feature-year vs
    target-year ambiguity that hid C6), the pre-registered refit strategy, the
    baseline-gate report, the paired significance test, and sensitivity results.

    Args:
        path: Path to write model_metadata.json.
        feature_names: Ordered list of feature names the model was trained on.
        target_code: WDI target indicator code.
        train_end: Last training year (feature years).
        val_end: Last validation year (feature years).
        metrics: Dictionary of evaluation metrics.
        model_type: Name of the model class.
        random_state: Random seed used.
        log_transform_features: Features that get log1p inside the pipeline.
        data_provenance: Panel path/hash/rows/countries/years/vintage dict.
        split_sizes: Row counts per split (train/val/test).
        split_target_years: Target-year windows per split (feature year + 1).
        refit_strategy: 'train_only' or 'train_val' (pre-registered).
        gate: Baseline-gate report from enforce_baseline_gate.
        significance: Paired bootstrap significance block.
        sensitivity: Sensitivity-analysis block (e.g. refit variants).
        extras: Additional top-level metadata keys (e.g. feature_selection).
    """
    metadata = {
        "target_code": target_code,
        "target_name": "GDP per capita growth (annual %)",
        "prediction_horizon_years": 1,
        "geographic_scope": "African countries",
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "log_transform_features": log_transform_features or [],
        "train_end": train_end,
        "val_end": val_end,
        "model_type": model_type,
        "random_state": random_state,
        "metrics": metrics,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit_or_none(),
        "library_versions": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "data_provenance": data_provenance,
        "split_sizes": split_sizes,
        "split_target_years": split_target_years or {
            "train": [2001, train_end + 1],
            "val": [train_end + 2, val_end + 1],
            "test": [val_end + 2, None],
        },
        "refit_strategy": refit_strategy,
        "gate": gate,
        "significance": significance,
        "sensitivity": sensitivity,
    }
    if extras:
        metadata.update(extras)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Model metadata written to %s", path)