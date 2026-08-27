"""Finalize model: leakage-free selection protocol with a baseline gate.

Pre-registered order of operations (C1/C4/C5/C6 remediation). The test split is
sealed until Step H and is never consulted for any selection decision:

    Step A  Load panel, build target, coverage-filter features on TRAIN mask.
    Step B  Temporal split; apply the B10 fair-comparison filter to test.
    Step C  Compute baselines on VALIDATION (this is what C1 lacked).
    Step D  Expanding-window hyperparameter search INSIDE the train period.
    Step E  Fit CV-best of each family on train, score on val, pick winner
            by validation MAE.
    Step F  Baseline gate: refuse to write artifacts unless the winner beats
            every validation baseline (override only via --allow-baseline-failure,
            which must then be disclosed in the report).
    Step G  Refit policy (pre-registered 'train_only'; 'train_val' recorded as
            sensitivity only — C6 explains why train+val refit is biased).
    Step H  Touch the test set ONCE: predictions, metrics, three baselines.
    Step I  Paired bootstrap significance vs the global-mean baseline.
    Step J  Write artifacts: pipeline, predictions, importance (on VAL),
            ridge coefficients, metadata with provenance/gate/significance.

Feature interpretation (permutation importance) is computed on VALIDATION, not
test (H1); direction comes from Ridge standardized coefficients, mapped with
get_transformed_feature_names, never zip(features, coef_) (H2).
"""
import argparse
import copy
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, Config
from src.evaluate import compute_permutation_importance_with_ci
from src.features import (
    build_feature_matrix,
    create_temporal_split,
    create_target,
    select_features_by_coverage,
)
from src.train import (
    build_hgb_pipeline,
    build_ridge_pipeline,
    compute_metrics,
    enforce_baseline_gate,
    get_transformed_feature_names,
    global_mean_baseline,
    persistence_baseline,
    save_pipeline,
    search_hyperparameters,
    train_and_evaluate,
    write_model_metadata,
    _sha256,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Pre-registered refit policy (Task 2.3 Step G / C6): validation targets
# (2019-2021) include the COVID crash (target mean -4.76) while test targets
# (2022-2024) do not; refitting on train+val injects a regime that does not
# represent the test period. The choice is made a priori, NOT by peeking at
# test results. 'train_val' is computed as a sensitivity analysis only.
REFIT_STRATEGY = "train_only"


def country_historical_mean_baseline(
    panel: pd.DataFrame,
    target_code: str,
    eval_rows: pd.DataFrame,
    fallback_value: float,
) -> np.ndarray:
    """Expanding country-mean growth up to each row's feature year (no future info).

    Spec section 10 optional baseline 3. For a row (country c, feature year t),
    the prediction is the mean of c's observed growth over all years <= t.
    Countries without any observed history fall back to the training global mean.

    Args:
        panel: Full country-year panel with the raw growth column.
        target_code: Column name of current-year growth (e.g. NY.GDP.PCAP.KD.ZG).
        eval_rows: Rows being scored, with iso3 and year columns.
        fallback_value: Value used when a country has no history <= t.

    Returns:
        Array of predictions aligned with eval_rows order.
    """
    hist = panel[["iso3", "year", target_code]].sort_values(["iso3", "year"]).copy()
    hist["hist_mean"] = (
        hist.groupby("iso3")[target_code]
        .transform(lambda s: s.expanding(min_periods=1).mean())
    )
    merged = eval_rows[["iso3", "year"]].merge(
        hist[["iso3", "year", "hist_mean"]], on=["iso3", "year"], how="left"
    )
    pred = merged["hist_mean"].fillna(fallback_value).to_numpy(dtype=float)
    n_fb = int(merged["hist_mean"].isna().sum())
    logger.info(
        "Country historical-mean baseline: expanding, n=%d (%d rows fell back to global mean)",
        len(pred), n_fb,
    )
    return pred


def main(
    config_path: Path = Path("config/indicators.yaml"),
    panel_path: Path = Path("data/processed/model_data.parquet"),
    output_dir: Path = Path("models"),
    data_dir: Optional[Path] = None,
    allow_baseline_failure: bool = False,
    wdi_vintage: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the full selection-finalization protocol and write artifacts.

    Args:
        config_path: YAML project config.
        panel_path: Processed country-year panel parquet.
        output_dir: Directory for model artifacts (models/ by default).
        data_dir: Directory for country_metadata.csv (defaults to panel parent).
        allow_baseline_failure: Ship despite a failed gate (must be disclosed).
        wdi_vintage: Free-text WDI vintage note for provenance.

    Returns:
        Summary dict with the metadata written to model_metadata.json.
    """
    config: Config = load_config(config_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(data_dir) if data_dir is not None else Path(panel_path).parent
    data_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- Step A: load, target, coverage-filter on train mask ----
    panel = pd.read_parquet(panel_path)
    panel = create_target(panel, config.target_code)

    feature_cols = [c for c in panel.columns if c not in
                    ["iso3", "country_name", "year", "target_next_year"]]
    train_mask = panel["year"] <= config.train_end
    panel = select_features_by_coverage(panel, feature_cols, min_coverage=0.6,
                                        train_mask=train_mask)
    final_features = [c for c in panel.columns if c not in
                      ["iso3", "country_name", "year", "target_next_year"]]
    log_features = [f for f in config.log_transform_candidates if f in final_features]

    # ---------------- Step B: split + B10 fair-comparison filter -------------
    train, val, test = create_temporal_split(panel, config.train_end, config.val_end)

    if config.target_code in test.columns:
        valid_persistence_mask = test[config.target_code].notna()
        n_dropped = int((~valid_persistence_mask).sum())
        if n_dropped > 0:
            logger.warning(
                "Dropping %d test rows with missing current-year growth "
                "(needed for fair persistence baseline comparison)", n_dropped)
            test = test[valid_persistence_mask]

    X_train, y_train = build_feature_matrix(train, final_features)
    X_val, y_val = build_feature_matrix(val, final_features)

    logger.info("Train: %s, Val: %s (test sealed until Step H)",
                X_train.shape, X_val.shape)

    # ---------------- Step C: baselines on VALIDATION (the C1 fix) ----------
    val_baselines = {
        "global_mean": compute_metrics(
            y_val.values, global_mean_baseline(y_train, len(y_val))),
    }
    pers_val = val.loc[y_val.index, config.target_code].values
    pers_ok = ~np.isnan(pers_val)
    if pers_ok.any():
        val_baselines["persistence"] = compute_metrics(
            y_val.values[pers_ok], pers_val[pers_ok])
    logger.info("Validation baselines: %s",
                {k: round(v["mae"], 4) for k, v in val_baselines.items()})

    # ---------------- Step D: expanding-window CV, train period ONLY ---------
    ridge_cv = search_hyperparameters(
        lambda **p: build_ridge_pipeline(
            log_transform_features=log_features,
            all_feature_names=final_features, **p),
        [{"alpha": a} for a in config.ridge_alpha_grid],
        X_train, y_train, train.loc[y_train.index, "year"],
        config.cv_initial_train_end, config.cv_val_window,
    )
    hgb_param_grid = [
        {"max_depth": d, "learning_rate": lr, "max_iter": mi}
        for d in config.hgb_grid["max_depth"]
        for lr in config.hgb_grid["learning_rate"]
        for mi in config.hgb_grid["max_iter"]
    ]
    hgb_cv = search_hyperparameters(
        lambda **p: build_hgb_pipeline(random_state=config.random_state, **p),
        hgb_param_grid, X_train, y_train,
        train.loc[y_train.index, "year"],
        config.cv_initial_train_end, config.cv_val_window,
    )
    ridge_cv.to_csv(output_dir / "cv_results_ridge.csv", index=False)
    hgb_cv.to_csv(output_dir / "cv_results_hgb.csv", index=False)
    logger.info("CV results written: ridge best %s | hgb best %s",
                ridge_cv.iloc[0].to_dict(), hgb_cv.iloc[0].to_dict())

    # ---------------- Step E: CV-best of each family -> validation ---------
    # Cast from the ranked frame (mixed dtypes upcast to float64 on read).
    ridge_best_params = {"alpha": float(ridge_cv.iloc[0]["alpha"])}
    hgb_best_params = {
        "max_depth": int(hgb_cv.iloc[0]["max_depth"]),
        "learning_rate": float(hgb_cv.iloc[0]["learning_rate"]),
        "max_iter": int(hgb_cv.iloc[0]["max_iter"]),
    }

    ridge_cv_best = build_ridge_pipeline(
        log_transform_features=log_features,
        all_feature_names=final_features, **ridge_best_params)
    hgb_cv_best = build_hgb_pipeline(
        random_state=config.random_state,
        l2_regularization=config.hgb_l2_regularization,
        early_stopping=config.hgb_early_stopping,
        validation_fraction=config.hgb_validation_fraction,
        n_iter_no_change=config.hgb_n_iter_no_change,
        **hgb_best_params)

    ridge_val_metrics = train_and_evaluate(ridge_cv_best, X_train, y_train, X_val, y_val)
    hgb_val_metrics = train_and_evaluate(hgb_cv_best, X_train, y_train, X_val, y_val)

    if hgb_val_metrics["mae"] <= ridge_val_metrics["mae"]:
        winner_name = "HistGradientBoostingRegressor"
        winner_pipeline = hgb_cv_best
        winner_val_metrics = hgb_val_metrics
        winner_family = "hgb"
    else:
        winner_name = "Ridge"
        winner_pipeline = ridge_cv_best
        winner_val_metrics = ridge_val_metrics
        winner_family = "ridge"
    logger.info("Winner by VALIDATION MAE: %s (%.4f) over Ridge (%.4f)",
                winner_name, winner_val_metrics["mae"], ridge_val_metrics["mae"])

    # ---------------- Step F: baseline gate (refuses artifact writes) -------
    gate = enforce_baseline_gate(winner_val_metrics, val_baselines, metric="mae")
    if not gate["passed"]:
        logger.error(
            "Refusing to write artifacts: selected model does not beat "
            "validation baselines. Failures: %s", gate["failures"])
        if not allow_baseline_failure:
            raise SystemExit(2)
        logger.warning(
            "--allow-baseline-failure set: writing artifacts for a model that "
            "FAILED the baseline gate. This must be disclosed in the report.")

    # ---------------- Step G: pre-registered refit policy -------------------
    if REFIT_STRATEGY == "train_only":
        winner_pipeline.fit(X_train, y_train)
    else:  # pragma: no cover - not the pre-registered path
        winner_pipeline.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))

    # Sensitivity analysis (NOT selection): same model refit on train+val.
    sensitivity: Dict[str, Any] = {"refit_train_val": None}

    # Feature interpretation on VALIDATION with the deployed (train-only) model
    # (Task 2.5: never reuse the sealed test split for interpretation).
    importance_df = compute_permutation_importance_with_ci(
        winner_pipeline, X_val, y_val, final_features,
        n_repeats=30, random_state=config.random_state)

    # Ridge coefficients for DIRECTION (spec section 13), mapped through
    # get_transformed_feature_names (H2: never zip(input order, coef_)).
    ridge_dir = build_ridge_pipeline(
        log_transform_features=log_features,
        all_feature_names=final_features, **ridge_best_params)
    ridge_dir.fit(X_train, y_train)
    coef_names = get_transformed_feature_names(ridge_dir, final_features)
    ridge_coefs = pd.DataFrame({
        "feature": coef_names,
        "coefficient": ridge_dir.named_steps["model"].coef_,
    })
    ridge_coefs["abs_coefficient"] = ridge_coefs["coefficient"].abs()
    ridge_coefs = ridge_coefs.sort_values("abs_coefficient", ascending=False)

    # ---------------- Step H: touch the test set EXACTLY ONCE ---------------
    X_test, y_test = build_feature_matrix(test, final_features)
    y_pred_test = winner_pipeline.predict(X_test)
    test_metrics = compute_metrics(y_test.values, y_pred_test)
    logger.info("Test metrics: %s", {k: round(v, 4) for k, v in test_metrics.items()})

    gm_pred = global_mean_baseline(y_train, len(y_test))
    gm_metrics = compute_metrics(y_test.values, gm_pred)

    test_with_target = test.loc[y_test.index]
    pers_growth = test_with_target[config.target_code].values
    valid_persistence = ~np.isnan(pers_growth)
    if valid_persistence.any():
        pers_metrics = compute_metrics(
            y_test.values[valid_persistence],
            persistence_baseline(pd.Series(pers_growth[valid_persistence])))
    else:
        pers_metrics = {}
        logger.warning("No valid persistence rows on test; baseline skipped")

    chm_pred = country_historical_mean_baseline(
        panel, config.target_code, test_with_target,
        fallback_value=float(y_train.mean()))
    chm_metrics = compute_metrics(y_test.values, chm_pred)
    logger.info("Test baselines - global mean %.4f | persistence %.4f | country hist-mean %.4f",
                gm_metrics["mae"],
                pers_metrics.get("mae", float("nan")),
                chm_metrics["mae"])

    # Sensitivity: refit on train+val variant, reported alongside (never used
    # to pick the primary strategy).
    alt_pipeline = copy.deepcopy(winner_pipeline)
    alt_pipeline.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
    y_pred_tv = alt_pipeline.predict(X_test)
    sensitivity["refit_train_val"] = {
        "test_metrics": compute_metrics(y_test.values, y_pred_tv),
        "mean_prediction": float(np.mean(y_pred_tv)),
        "mean_actual": float(np.mean(y_test.values)),
        "bias_note": ("train+val refit absorbs the 2020 COVID crash (validation "
                      "target mean is far below test target mean), depressing "
                      "central tendency on test - the C6 mechanism"),
    }

    # ---------------- Step I: paired bootstrap significance -----------------
    resid_model = np.abs(y_test.values - y_pred_test)
    resid_base = np.abs(y_test.values - gm_pred)
    diff = resid_base - resid_model            # positive => model better
    rng = np.random.RandomState(config.random_state)
    boot = np.array([diff[rng.randint(0, len(diff), len(diff))].mean()
                     for _ in range(5000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    significance = {
        "paired_mae_improvement_vs_global_mean": float(diff.mean()),
        "ci_lower": float(lo),
        "ci_upper": float(hi),
        "significant_at_95": bool(lo > 0),
        "n_bootstrap": 5000,
    }
    logger.info("Paired MAE improvement %.4f pp, 95%% CI [%.4f, %.4f], significant=%s",
                diff.mean(), lo, hi, lo > 0)

    # ---------------- Step J: write artifacts -------------------------------
    save_pipeline(winner_pipeline, output_dir / "growth_model.joblib")

    test_predictions = pd.DataFrame({
        "iso3": test_with_target["iso3"].values,
        "year": test_with_target["year"].values,
        "country_name": test_with_target["country_name"].values,
        "actual": y_test.values,
        "predicted": y_pred_test,
    })
    test_predictions.to_parquet(output_dir / "test_predictions.parquet", index=False)
    importance_df.to_parquet(output_dir / "feature_importance.parquet", index=False)
    ridge_coefs.to_parquet(output_dir / "ridge_coefficients.parquet", index=False)

    all_metrics = {
        "global_mean_baseline": gm_metrics,
        "persistence_baseline": pers_metrics,
        "country_historical_mean_baseline": chm_metrics,
        "ridge_val": ridge_val_metrics,
        "hgb_val": hgb_val_metrics,
        "val_baselines": val_baselines,
        "winner_val": winner_val_metrics,
        "winner_test": test_metrics,
    }

    train_years = train.loc[y_train.index, "year"]
    val_years = val.loc[y_val.index, "year"]
    test_years = test.loc[y_test.index, "year"]
    panel_path = Path(panel_path)
    provenance = {
        "panel_path": str(panel_path),
        "panel_sha256": _sha256(panel_path),
        "panel_rows": int(len(panel)),
        "n_countries": int(panel["iso3"].nunique()),
        "year_min": int(panel["year"].min()),
        "year_max": int(panel["year"].max()),
        "wdi_vintage": wdi_vintage or "unrecorded (WDI bulk download predating commit)",
    }

    write_model_metadata(
        path=output_dir / "model_metadata.json",
        feature_names=final_features,
        target_code=config.target_code,
        train_end=config.train_end,
        val_end=config.val_end,
        metrics=all_metrics,
        model_type=winner_name,
        random_state=config.random_state,
        log_transform_features=log_features,
        data_provenance=provenance,
        split_sizes={"train": int(len(y_train)), "val": int(len(y_val)),
                     "test": int(len(y_test))},
        split_target_years={
            "train": [int(train_years.min()) + 1, int(train_years.max()) + 1],
            "val": [int(val_years.min()) + 1, int(val_years.max()) + 1],
            "test": [int(test_years.min()) + 1, int(test_years.max()) + 1],
        },
        refit_strategy=REFIT_STRATEGY,
        gate=gate,
        significance=significance,
        sensitivity=sensitivity,
    )

    country_meta = panel[["iso3", "country_name"]].drop_duplicates().sort_values("iso3")
    country_meta.to_csv(data_dir / "country_metadata.csv", index=False)
    logger.info("Country metadata saved (%d countries)", len(country_meta))

    logger.info("All artifacts written to %s", output_dir)
    return json.loads((output_dir / "model_metadata.json").read_text(encoding="utf-8"))


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to sys.argv[1:]).

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("config/indicators.yaml"))
    parser.add_argument("--panel", type=Path,
                        default=Path("data/processed/model_data.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--wdi-vintage", type=str, default=None,
                        help="Free-text WDI vintage note recorded in metadata")
    parser.add_argument("--allow-baseline-failure", action="store_true",
                        help="Ship a model that FAILED the baseline gate; the "
                             "failure is recorded in metadata and MUST be "
                             "disclosed in the report. Never use to quietly "
                             "bypass the gate.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    main(
        config_path=args.config,
        panel_path=args.panel,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        allow_baseline_failure=args.allow_baseline_failure,
        wdi_vintage=args.wdi_vintage,
    )
