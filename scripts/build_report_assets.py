"""Generate every number any document may cite, from committed artifacts only.

C3 remediation: the previous report contained hand-typed statistics that did not
match the data. This script removes the possibility by construction — it loads
`models/model_metadata.json`, `models/test_predictions.parquet`,
`models/feature_importance.parquet`, `models/ridge_coefficients.parquet`,
`models/cv_results_*.csv` and `data/processed/model_data.parquet`, and emits:

    reports/generated/metrics.json          every citable number
    reports/generated/table_*.md            ready-to-paste Markdown tables
    figures/*.png                            report figures (project figures folder)

Documents may then quote ONLY values present in metrics.json / the generated
tables (tests/test_report_assets.py enforces this for MAE figures and worst-
error country names). Deterministic: same artifacts in, same bytes out.
"""
import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # headless rendering for report figures

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from src.config import load_config
from src.evaluate import compute_metrics_by_group, compute_worst_errors
from src.visualization import (
    get_project_palette,
    plot_correlation_matrix,
    plot_actual_vs_predicted,
    plot_feature_importance,
    plot_residuals,
    set_project_style,
)

logger = logging.getLogger(__name__)

# Pairs the report prose quotes explicitly; recomputed here, never typed.
KEY_CORRELATIONS: Dict[str, Tuple[str, str]] = {
    "electricity_vs_internet": ("EG.ELC.ACCS.ZS", "IT.NET.USER.ZS"),
    "gdppc_vs_life_expectancy": ("NY.GDP.PCAP.CD", "SP.DYN.LE00.IN"),
    "inflation_vs_growth": ("FP.CPI.TOTL.ZG", "NY.GDP.PCAP.KD.ZG"),
    "capital_formation_vs_growth": ("NE.GDI.TOTL.ZS", "NY.GDP.PCAP.KD.ZG"),
}


def _fmt(v: float, places: int = 2) -> str:
    """Format a float for Markdown tables with fixed precision.

    Args:
        v: Value to format (NaN-safe).
        places: Decimal places.

    Returns:
        Formatted string or 'n/a'.
    """
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"{v:.{places}f}"


def dataframe_to_markdown(df: pd.DataFrame, float_places: int = 2) -> str:
    """Render a DataFrame as a GitHub-flavored Markdown table.

    Args:
        df: Table to render.
        float_places: Decimal places for float columns.

    Returns:
        Markdown string.
    """
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda v: _fmt(float(v), float_places))
    header = "| " + " | ".join(str(c) for c in out.columns) + " |"
    sep = "|" + "|".join(["---"] * len(out.columns)) + "|"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in out.itertuples(index=False)]
    return "\n".join([header, sep] + rows)


def load_artifacts(root: Path) -> Dict[str, Any]:
    """Load every committed artifact the generator is allowed to use.

    Args:
        root: Repository root.

    Returns:
        Dict of metadata, predictions, importance, coefficients, CV tables,
        panel, and config.

    Raises:
        FileNotFoundError: If any required artifact is missing.
    """
    paths = {
        "metadata": root / "models/model_metadata.json",
        "test_predictions": root / "models/test_predictions.parquet",
        "feature_importance": root / "models/feature_importance.parquet",
        "ridge_coefficients": root / "models/ridge_coefficients.parquet",
        "cv_ridge": root / "models/cv_results_ridge.csv",
        "cv_hgb": root / "models/cv_results_hgb.csv",
        "panel": root / "data/processed/model_data.parquet",
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing artifacts: {missing}. Run scripts/finalize_model.py first.")
    art: Dict[str, Any] = {}
    art["metadata"] = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    art["test_predictions"] = pd.read_parquet(paths["test_predictions"])
    art["feature_importance"] = pd.read_parquet(paths["feature_importance"])
    art["ridge_coefficients"] = pd.read_parquet(paths["ridge_coefficients"])
    art["cv_ridge"] = pd.read_csv(paths["cv_ridge"])
    art["cv_hgb"] = pd.read_csv(paths["cv_hgb"])
    art["panel"] = pd.read_parquet(paths["panel"])
    art["config"] = load_config(root / "config/indicators.yaml")
    return art


def build_model_comparison_table(art: Dict[str, Any]) -> pd.DataFrame:
    """Assemble the baselines + families comparison table (val and test).

    Args:
        art: Loaded artifacts dict.

    Returns:
        DataFrame for markdown rendering.
    """
    mm = art["metadata"]["metrics"]
    name = art["metadata"]["model_type"]

    def row(label: str, split: str, m: Dict[str, float]) -> Dict[str, Any]:
        return {
            "Model": label, "Split": split,
            "MAE": m.get("mae"), "RMSE": m.get("rmse"), "R2": m.get("r2"),
            "Dir. acc": m.get("directional_accuracy"),
            "Majority rate": m.get("directional_majority_rate"),
            "Dir. skill": m.get("directional_skill"),
        }

    rows = [
        row("Global mean baseline", "test", mm["global_mean_baseline"]),
        row("Persistence baseline", "test", mm.get("persistence_baseline", {})),
        row("Country historical mean baseline", "test",
            mm.get("country_historical_mean_baseline", {})),
        row(f"{name} (deployed)", "test", mm["winner_test"]),
        row("Global mean baseline", "validation", mm["val_baselines"]["global_mean"]),
        row("Persistence baseline", "validation", mm["val_baselines"]["persistence"]),
        row("Ridge (CV-best)", "validation", mm["ridge_val"]),
        row("HGB (CV-best)", "validation", mm["hgb_val"]),
    ]
    return pd.DataFrame(rows)


def build_eda_summary(art: Dict[str, Any]) -> pd.DataFrame:
    """Per-feature min / median / max / train coverage, computed from the panel.

    Args:
        art: Loaded artifacts dict.

    Returns:
        Summary table in feature order.
    """
    panel, cfg = art["panel"], art["config"]
    features = [f.code for f in cfg.features if f.code in panel.columns]
    train_rows = panel[panel["year"] <= cfg.train_end]
    rows = []
    for f in features:
        s = panel[f].dropna()
        cov_train = 100 * train_rows[f].notna().mean()
        rows.append({"feature": f, "n": int(s.size), "min": s.min(), "median": s.median(),
                     "max": s.max(), "train_coverage_pct": cov_train})
    target = panel[cfg.target_code].dropna()
    rows.append({"feature": cfg.target_code + " (target)", "n": int(target.size),
                 "min": target.min(), "median": target.median(), "max": target.max(),
                 "train_coverage_pct": 100 * train_rows[cfg.target_code].notna().mean()})
    return pd.DataFrame(rows)


def build_correlation_tables(art: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Compute the quoted key correlations and the strongest pairwise ones.

    Args:
        art: Loaded artifacts dict.

    Returns:
        (table of all pairs with |r| >= 0.6 sorted descending,
         dict of the four report-quoted key correlations)
    """
    panel = art["panel"]
    cfg = art["config"]
    features = [f.code for f in cfg.features if f.code in panel.columns]
    corr = panel[features + [cfg.target_code]].corr()

    key_vals = {}
    for label, (a, b) in KEY_CORRELATIONS.items():
        if a in corr.columns and b in corr.columns:
            key_vals[label] = float(corr.loc[a, b])

    pairs = []
    for i, a in enumerate(corr.columns):
        for b in corr.columns[i + 1:]:
            r = corr.loc[a, b]
            if pd.notna(r) and abs(r) >= 0.6:
                pairs.append({"pair": f"{a} vs {b}", "pearson_r": float(r)})
    table = pd.DataFrame(pairs).sort_values("pearson_r", key=abs, ascending=False)
    return table, key_vals


def build_yearly_table(art: Dict[str, Any]) -> pd.DataFrame:
    """Test-set metrics per target year, recomputed from the frozen parquet.

    Args:
        art: Loaded artifacts dict.

    Returns:
        Yearly metrics table.
    """
    preds = art["test_predictions"]
    by_year = compute_metrics_by_group(preds, group_col="year")
    cols = ["year", "mae", "rmse", "r2", "directional_accuracy",
            "directional_majority_rate", "directional_skill"]
    by_year["n"] = preds.groupby("year").size().values
    return by_year[[c for c in cols if c in by_year.columns]].rename(columns={
        "mae": "MAE", "rmse": "RMSE", "r2": "R2",
        "directional_accuracy": "Dir. acc", "directional_majority_rate": "Majority rate",
        "directional_skill": "Dir. skill"})


def build_worst_errors_table(art: Dict[str, Any]) -> pd.DataFrame:
    """Top-10 absolute errors from the deployed model's frozen test predictions.

    Args:
        art: Loaded artifacts dict.

    Returns:
        Worst-error rows with country, year, actual, predicted, abs error.
    """
    worst = compute_worst_errors(art["test_predictions"], top_n=10)
    out = worst[["country_name", "year", "actual", "predicted", "abs_error"]].rename(
        columns={"country_name": "Country", "year": "Year", "actual": "Actual",
                 "predicted": "Predicted", "abs_error": "Abs error (pp)"})
    return out.reset_index(drop=True)


def build_feature_tables(art: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Importance and Ridge-coefficient tables with readable feature names.

    Args:
        art: Loaded artifacts dict.

    Returns:
        (importance table, ridge coefficient table)
    """
    name_of = {f.code: f.name for f in art["config"].features}
    name_of[art["config"].target_code] = "GDP per capita growth (annual %), current year"
    imp = art["feature_importance"].copy()
    imp.insert(1, "name", imp["feature"].map(name_of).fillna(imp["feature"]))
    imp = imp[["feature", "name", "importance_mean", "importance_std",
               "ci_lower", "ci_upper", "is_significant"]]
    imp["is_significant"] = imp["is_significant"].map({True: "yes", False: "noise"})

    rc = art["ridge_coefficients"].copy()
    rc["display"] = rc["feature"].map(
        lambda s: name_of.get(s.replace("_log1p", ""), s.replace("_log1p", ""))
        + (" (log1p)" if s.endswith("_log1p") else ""))
    rc = rc[["feature", "display", "coefficient"]].rename(columns={"display": "name"})
    return imp, rc


def build_cv_table(art: Dict[str, Any]) -> pd.DataFrame:
    """Top CV rows for both families into one ranked table.

    Args:
        art: Loaded artifacts dict.

    Returns:
        Combined CV results table (best first).
    """
    rows = []
    for family, df in (("Ridge", art["cv_ridge"]), ("HGB", art["cv_hgb"])):
        d = df.copy()
        param_cols = [c for c in d.columns if c not in ("mean_mae", "std_mae", "n_folds")]
        d["config"] = family + ": " + d[param_cols].apply(
            lambda r: ", ".join(f"{c}={v:g}" for c, v in r.items()), axis=1)
        for _, r in d.head(3).iterrows():
            rows.append({"config": r["config"], "mean fold MAE": r["mean_mae"],
                         "std": r["std_mae"], "folds": int(r["n_folds"])})
    return pd.DataFrame(rows).sort_values("mean fold MAE").reset_index(drop=True)


def build_splits_table(art: Dict[str, Any]) -> pd.DataFrame:
    """Feature-year vs target-year split table from metadata (C6 clarity).

    Args:
        art: Loaded artifacts dict.

    Returns:
        Split definition table.
    """
    meta = art["metadata"]
    st, ss = meta["split_target_years"], meta["split_sizes"]
    rows = []
    for split in ("train", "val", "test"):
        lo, hi = st[split]
        rows.append({"split": split, "feature_years": f"{lo - 1}\u2013{hi - 1}",
                     "target_years": f"{lo}\u2013{hi}", "n": ss[split]})
    return pd.DataFrame(rows)


def build_trend_stats(art: Dict[str, Any]) -> Dict[str, float]:
    """Growth-trend summary numbers for the EDA narrative.

    Args:
        art: Loaded artifacts dict.

    Returns:
        Dict of computed trend statistics.
    """
    panel = art["panel"]
    cfg = art["config"]
    g = panel[["year", cfg.target_code]].dropna()

    def window(a: int, b: int) -> float:
        return float(g.loc[g["year"].between(a, b), cfg.target_code].mean())

    med_2020 = float(g.loc[g["year"] == 2020, cfg.target_code].median())
    med_2021_22 = float(g.loc[g["year"].between(2021, 2022), cfg.target_code].median())
    return {
        "mean_growth_2000_2010": window(2000, 2010),
        "mean_growth_2011_2019": window(2011, 2019),
        "mean_growth_2020_2024": window(2020, 2024),
        "median_growth_2020": med_2020,
        "median_growth_2021_2022": med_2021_22,
    }


def build_feature_selection_table(art: Dict[str, Any]) -> pd.DataFrame:
    """Coverage-filter outcome recorded by finalize (kept/dropped candidates).

    Args:
        art: Loaded artifacts dict.

    Returns:
        Table of dropped candidates with their training coverage, plus kept count.
    """
    fs = art["metadata"].get("feature_selection", {})
    rows = [{"feature": c, "train_coverage_pct": cov, "decision": "dropped"}
            for c, cov in (fs.get("dropped") or {}).items()]
    for c in art["metadata"]["feature_names"]:
        rows.append({"feature": c, "train_coverage_pct": float("nan"),
                     "decision": f"kept (>= {fs.get('min_coverage_pct', 60.0)}% required)"})
    return pd.DataFrame(rows)


def build_figures(art: Dict[str, Any], fig_dir: Path) -> List[str]:
    """Render report figures with the project visual language.

    Args:
        art: Loaded artifacts dict.
        fig_dir: Output directory for PNGs.

    Returns:
        Written file names.
    """
    fig_dir.mkdir(parents=True, exist_ok=True)
    set_project_style()
    preds = art["test_predictions"]
    written = []

    fig = plot_actual_vs_predicted(preds["actual"].to_numpy(), preds["predicted"].to_numpy(),
                                   title="Test: actual vs predicted next-year growth")
    fig.savefig(fig_dir / "actual_vs_predicted.png", dpi=110)
    plt.close(fig)
    written.append("actual_vs_predicted.png")

    fig = plot_residuals(preds["actual"].to_numpy(), preds["predicted"].to_numpy(),
                         title="Test: residuals vs predicted")
    fig.savefig(fig_dir / "residuals.png", dpi=110)
    plt.close(fig)
    written.append("residuals.png")

    imp = art["feature_importance"]
    sig = imp[imp["is_significant"]]
    series_src = sig if not sig.empty else imp
    series = pd.Series(series_src["importance_mean"].to_numpy(),
                       index=series_src["feature"].to_numpy())
    fig = plot_feature_importance(
        series, title="Permutation importance (validation): CI-excludes-zero features")
    fig.savefig(fig_dir / "feature_importance.png", dpi=110)
    plt.close(fig)
    written.append("feature_importance.png")

    cfg = art["config"]
    feats = [f.code for f in cfg.features if f.code in art["panel"].columns]
    # Same diverging colormap as the application: no rainbow, no default RdBu.
    fig = plot_correlation_matrix(
        art["panel"], feats, title="Feature correlations (committed panel)")
    fig.set_size_inches(11, 9)
    fig.tight_layout()
    fig.savefig(fig_dir / "correlation_heatmap.png", dpi=110)
    plt.close(fig)
    written.append("correlation_heatmap.png")

    logger.info("Figures written: %s", ", ".join(written))
    return written


def compute_test_bootstrap_cis(art: Dict[str, Any]) -> Dict[str, Any]:
    """Bootstrap CIs for the deployed model's test metrics (deterministic, seed 42).

    Args:
        art: Loaded artifacts dict.

    Returns:
        Dict with mae/rmse interval blocks computed from test_predictions.
    """
    from src.evaluate import compute_bootstrap_ci
    preds = art["test_predictions"]
    actual, pred = preds["actual"].to_numpy(), preds["predicted"].to_numpy()
    mae_lo, mae_hi = compute_bootstrap_ci(
        actual, pred, lambda a, p: float(np.mean(np.abs(a - p))), n_bootstrap=2000)
    rmse_lo, rmse_hi = compute_bootstrap_ci(
        actual, pred, lambda a, p: float(np.sqrt(np.mean((a - p) ** 2))), n_bootstrap=2000)
    return {
        "n_bootstrap": 2000,
        "mae_lower": float(mae_lo), "mae_upper": float(mae_hi),
        "rmse_lower": float(rmse_lo), "rmse_upper": float(rmse_hi),
        "mae_ci": [float(mae_lo), float(mae_hi)],
        "rmse_ci": [float(rmse_lo), float(rmse_hi)],
    }


def assemble_metrics(art: Dict[str, Any]) -> Dict[str, Any]:
    """Build the citable-numbers registry (reports/generated/metrics.json).

    Args:
        art: Loaded artifacts dict.

    Returns:
        Metrics dict: every number documents may quote.
    """
    meta = art["metadata"]
    mm = meta["metrics"]
    panel = art["panel"]
    cfg = art["config"]
    target = panel[cfg.target_code].dropna()
    corr_table, key_corr = build_correlation_tables(art)
    worst_raw = compute_worst_errors(art["test_predictions"], top_n=10)

    metrics: Dict[str, Any] = {
        "model_type": meta["model_type"],
        "n_features": meta["n_features"],
        "feature_names": meta["feature_names"],
        "refit_strategy": meta["refit_strategy"],
        "winner_test": mm["winner_test"],
        "global_mean_baseline": mm["global_mean_baseline"],
        "persistence_baseline": mm.get("persistence_baseline", {}),
        "country_historical_mean_baseline": mm.get("country_historical_mean_baseline", {}),
        "winner_val": mm["winner_val"],
        "ridge_val": mm["ridge_val"],
        "hgb_val": mm["hgb_val"],
        "val_baselines": mm["val_baselines"],
        "gate": meta["gate"],
        "significance": meta["significance"],
        "sensitivity": meta["sensitivity"],
        "split_sizes": meta["split_sizes"],
        "split_target_years": meta["split_target_years"],
        "data_provenance": meta["data_provenance"],
        "library_versions": meta["library_versions"],
        "worst_errors": [
            {k: (float(v) if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool) else v)
             for k, v in r.items()}
            for r in worst_raw.to_dict(orient="records")
        ],
        "target_stats": {
            "n": int(target.size), "min": float(target.min()), "max": float(target.max()),
            "mean": float(target.mean()), "median": float(target.median()),
            "pct_positive": float((target >= 0).mean()),
        },
        "correlations": key_corr,
        "trends": build_trend_stats(art),
        "test_bootstrap_cis": compute_test_bootstrap_cis(art),
        "importance_n_significant": int(art["feature_importance"]["is_significant"].sum()),
        "importance_n_total": int(len(art["feature_importance"])),
        "cv_best_ridge_mean_fold_mae": float(art["cv_ridge"].iloc[0]["mean_mae"]),
        "cv_best_hgb_mean_fold_mae": float(art["cv_hgb"].iloc[0]["mean_mae"]),
        "feature_selection": meta.get("feature_selection", {}),
        "deployed_hgb_iters": None,
    }

    # Deployed-model introspection (C5 evidence): n_iter_ vs max_iter.
    try:
        import joblib
        pipe = joblib.load(art["model_path"])
        if hasattr(pipe, "named_steps") and "model" in pipe.named_steps:
            mdl = pipe.named_steps["model"]
            if hasattr(mdl, "n_iter_"):
                metrics["deployed_hgb_iters"] = {"n_iter": int(mdl.n_iter_),
                                                 "max_iter": int(mdl.max_iter)}
    except Exception as e:  # introspection is optional; never break the generator
        logger.warning("Could not introspect deployed pipeline: %s", e)

    return metrics


def main(root: Optional[Path] = None,
         out_dir: Optional[Path] = None) -> Dict[str, Path]:
    """Generate all report assets from committed artifacts.

    Args:
        root: Repository root (default: cwd).
        out_dir: Output directory (default: <root>/reports/generated).

    Returns:
        Mapping of asset name to written path.
    """
    root = Path(root) if root else Path.cwd()
    out_dir = Path(out_dir) if out_dir else root / "reports" / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = root / "figures"

    art = load_artifacts(root)
    art["model_path"] = root / "models/growth_model.joblib"

    imp_tbl, rc_tbl = build_feature_tables(art)
    tables = {
        "table_model_comparison.md": build_model_comparison_table(art),
        "table_feature_importance.md": imp_tbl,
        "table_ridge_coefficients.md": rc_tbl,
        "table_yearly_metrics.md": build_yearly_table(art),
        "table_worst_errors.md": build_worst_errors_table(art),
        "table_eda_summary.md": build_eda_summary(art),
        "table_correlations.md": build_correlation_tables(art)[0],
        "table_cv_results.md": build_cv_table(art),
        "table_splits.md": build_splits_table(art),
        "table_feature_selection.md": build_feature_selection_table(art),
    }
    places_by_table = {"table_feature_importance.md": 3, "table_ridge_coefficients.md": 3}
    written: Dict[str, Path] = {}
    for name, df in tables.items():
        path = out_dir / name
        path.write_text(dataframe_to_markdown(df, float_places=places_by_table.get(name, 2)) + "\n",
                        encoding="utf-8")
        written[name] = path
        logger.info("Wrote %s (%d rows)", path.name, len(df))

    metrics = assemble_metrics(art)
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=False) + "\n",
                            encoding="utf-8")
    written["metrics.json"] = metrics_path
    logger.info("Wrote metrics.json with %d top-level keys", len(metrics))

    for f in build_figures(art, fig_dir):
        written[f] = fig_dir / f

    return written


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Build report assets from artifacts")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    main(root=args.root)
