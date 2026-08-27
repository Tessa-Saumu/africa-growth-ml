"""Finalize model: select winner, refit on train+val, evaluate on test, save artifacts.

This script runs after initial model comparison. It:
1. Loads processed data
2. Trains all candidate models on training data
3. Selects winner by validation MAE
4. Refits winner on train+val combined
5. Evaluates once on test (final reported metrics)
6. Saves: pipeline -> growth_model.joblib, metadata -> model_metadata.json
7. Precomputes permutation importance for the app
"""
import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.features import create_target, select_features_by_coverage, build_feature_matrix, create_temporal_split
from src.train import (
    build_ridge_pipeline, build_hgb_pipeline, train_and_evaluate,
    save_pipeline, compute_metrics, write_model_metadata, global_mean_baseline,
    persistence_baseline,
)
from src.evaluate import compute_permutation_importance

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    config = load_config()

    # Load processed data
    panel = pd.read_parquet("data/processed/model_data.parquet")
    panel = create_target(panel, config.target_code)

    # Feature selection on training data only
    feature_cols = [c for c in panel.columns if c not in
                   ["iso3", "country_name", "year", "target_next_year"]]
    train_mask = panel["year"] <= config.train_end
    panel = select_features_by_coverage(panel, feature_cols, min_coverage=0.6,
                                        train_mask=train_mask)

    # Determine final feature list
    final_features = [c for c in panel.columns if c not in
                     ["iso3", "country_name", "year", "target_next_year"]]

    # Split
    train, val, test = create_temporal_split(panel, config.train_end, config.val_end)

    # B10 FIX: Drop test rows with missing current-year growth globally.
    # Persistence baseline needs this value; the ML model also uses it as a feature.
    # If missing, baseline can't predict but ML can (via imputer), creating unfair
    # comparison (ML evaluated on more rows). Drop globally so all models are
    # evaluated on identical test observations.
    if "NY.GDP.PCAP.KD.ZG" in test.columns:
        valid_persistence_mask = test["NY.GDP.PCAP.KD.ZG"].notna()
        n_dropped = (~valid_persistence_mask).sum()
        if n_dropped > 0:
            logger.warning(
                "Dropping %d test rows with missing current-year growth "
                "(needed for fair persistence baseline comparison)", n_dropped
            )
            test = test[valid_persistence_mask]

    X_train, y_train = build_feature_matrix(train, final_features)
    X_val, y_val = build_feature_matrix(val, final_features)
    X_test, y_test = build_feature_matrix(test, final_features)

    logger.info("Train: %s, Val: %s, Test: %s", X_train.shape, X_val.shape, X_test.shape)

    # Determine log transform features from config
    log_features = [f for f in config.log_transform_candidates if f in final_features]

    # Train candidates
    ridge = build_ridge_pipeline(alpha=config.ridge_alpha,
                                 log_transform_features=log_features,
                                 all_feature_names=final_features)
    hgb = build_hgb_pipeline(max_iter=config.hgb_max_iter,
                             learning_rate=config.hgb_learning_rate,
                             max_depth=config.hgb_max_depth,
                             random_state=config.random_state)

    ridge_metrics = train_and_evaluate(ridge, X_train, y_train, X_val, y_val)
    hgb_metrics = train_and_evaluate(hgb, X_train, y_train, X_val, y_val)

    logger.info("Ridge val MAE: %.4f", ridge_metrics["mae"])
    logger.info("HGB val MAE: %.4f", hgb_metrics["mae"])

    # Select winner
    if hgb_metrics["mae"] <= ridge_metrics["mae"]:
        winner_name = "HistGradientBoostingRegressor"
        winner_pipeline = hgb
        winner_val_metrics = hgb_metrics
    else:
        winner_name = "Ridge"
        winner_pipeline = ridge
        winner_val_metrics = ridge_metrics

    logger.info("Winner: %s (val MAE: %.4f)", winner_name, winner_val_metrics["mae"])

    # Refit winner on train+val
    X_trainval = pd.concat([X_train, X_val])
    y_trainval = pd.concat([y_train, y_val])
    winner_pipeline.fit(X_trainval, y_trainval)

    # Evaluate on test (final metrics)
    y_pred_test = winner_pipeline.predict(X_test)
    test_metrics = compute_metrics(y_test.values, y_pred_test)
    logger.info("Test metrics: %s", test_metrics)

    # Baselines for comparison
    gm_pred = global_mean_baseline(y_train, len(y_test))
    gm_metrics = compute_metrics(y_test.values, gm_pred)
    logger.info("Global mean baseline test MAE: %.4f", gm_metrics["mae"])

    # Persistence: for test set, prediction = current year's growth
    # Use the same test rows that have valid target (y_test is already filtered)
    # Get the current year growth for those rows
    test_with_target = test.loc[y_test.index]
    persistence_growth = test_with_target["NY.GDP.PCAP.KD.ZG"].values
    valid_persistence = ~np.isnan(persistence_growth)
    if valid_persistence.any():
        pers_pred = persistence_baseline(pd.Series(persistence_growth[valid_persistence]))
        pers_metrics = compute_metrics(
            y_test.values[valid_persistence], pers_pred
        )
        logger.info("Persistence baseline test MAE: %.4f", pers_metrics["mae"])
    else:
        pers_metrics = {}

    # Save pipeline
    save_pipeline(winner_pipeline, Path("models/growth_model.joblib"))

    # Precompute permutation importance for the app
    importance = compute_permutation_importance(
        winner_pipeline, X_test, y_test, final_features, n_repeats=10
    )

    # B11 FIX: Save precomputed test predictions as parquet.
    # The app reads this file instead of calling model.predict on every rerun,
    # which would lag badly on Streamlit Cloud's small instances.
    # Use test rows that have valid target (same index as y_test)
    test_with_target = test.loc[y_test.index]
    test_predictions = pd.DataFrame({
        "iso3": test_with_target["iso3"].values,
        "year": test_with_target["year"].values,
        "country_name": test_with_target["country_name"].values,
        "actual": y_test.values,
        "predicted": y_pred_test,
    })
    test_predictions.to_parquet("models/test_predictions.parquet", index=False)
    logger.info("Test predictions saved to models/test_predictions.parquet")

    # B11 FIX: Save precomputed permutation importance as parquet too.
    importance_df = importance.reset_index()
    importance_df.columns = ["feature", "importance"]
    importance_df.to_parquet("models/feature_importance.parquet", index=False)
    logger.info("Feature importance saved to models/feature_importance.parquet")

    # Write metadata
    all_metrics = {
        "global_mean_baseline": gm_metrics,
        "persistence_baseline": pers_metrics if valid_persistence.any() else {},
        "ridge_val": ridge_metrics,
        "hgb_val": hgb_metrics,
        "winner_test": test_metrics,
        "feature_importance": importance.to_dict(),
    }

    write_model_metadata(
        path=Path("models/model_metadata.json"),
        feature_names=final_features,
        target_code=config.target_code,
        train_end=config.train_end,
        val_end=config.val_end,
        metrics=all_metrics,
        model_type=winner_name,
        random_state=config.random_state,
        log_transform_features=log_features,
    )

    # Save country metadata
    country_meta = panel[["iso3", "country_name"]].drop_duplicates().sort_values("iso3")
    country_meta.to_csv("data/processed/country_metadata.csv", index=False)
    logger.info("Country metadata saved (%d countries)", len(country_meta))

    logger.info("All artifacts saved. Ready for deployment.")


if __name__ == "__main__":
    main()