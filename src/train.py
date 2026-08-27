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
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
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
    max_iter: int = 1000,
    learning_rate: float = 0.05,
    max_depth: int = 5,
    random_state: int = 42,
) -> Pipeline:
    """Build HistGradientBoostingRegressor pipeline with imputation.

    Args:
        max_iter: Maximum boosting iterations.
        learning_rate: Boosting learning rate.
        max_depth: Maximum tree depth.
        random_state: Random seed for reproducibility.

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
        )),
    ])
    logger.info(
        "Built HGB pipeline (max_iter=%d, lr=%.4f, depth=%d)",
        max_iter, learning_rate, max_depth
    )
    return pipeline


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
) -> None:
    """Write model metadata JSON with feature contract and metrics.

    B5 FIX: This file is the single source of truth for the model's feature
    contract. The app reads feature names from here, not from config.

    Args:
        path: Path to write model_metadata.json.
        feature_names: Ordered list of feature names the model was trained on.
        target_code: WDI target indicator code.
        train_end: Last training year.
        val_end: Last validation year.
        metrics: Dictionary of evaluation metrics.
        model_type: Name of the model class.
        random_state: Random seed used.
        log_transform_features: Features that get log1p inside the pipeline.
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
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Model metadata written to %s", path)