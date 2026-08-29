"""Africa Growth Explorer - Streamlit application.

Decision-support system for predicting near-term GDP per capita growth
across African countries using World Bank Development Indicators.

This application loads a pre-trained model and provides:
- Project Overview with the causal guardrail
- Explore Africa: country-level indicator trends and comparisons
- Model Performance: metrics, actual vs predicted, feature importance
- Scenario Explorer: interactive what-if predictions with extrapolation
  guardrails

Presentation follows the Terracotta Editorial design system in
``src/theme.py`` and ``src/ui.py``. The analytical logic (feature selection,
guardrails, significance interpretation, scenario responsiveness) is
unchanged by that layer.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from src import ui
from src.theme import COLORS, inject_editorial_styles
from src.visualization import (
    set_editorial_plot_style,
    get_project_palette,
    coverage_points,
    load_africa_coordinates,
    load_africa_geometry,
    plot_actual_vs_predicted,
    plot_africa_map,
    plot_feature_importance,
    plot_growth_trend,
    plot_indicator_small_multiples,
    plot_metric_by_year,
    plot_residuals,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Apply project visual style (must be before any matplotlib usage)
set_editorial_plot_style()
PALETTE = get_project_palette()

# Set matplotlib backend for Streamlit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Page configuration
FAVICON = Path("static/favicon.png")
st.set_page_config(
    page_title="Africa Growth Explorer",
    page_icon=str(FAVICON) if FAVICON.exists() else None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Type alias for sklearn pipeline
from sklearn.pipeline import Pipeline as SklearnPipeline

PAGES = [
    "Project Overview",
    "Explore Africa",
    "Model Performance",
    "Scenario Explorer",
]

# Indicators used for the country profile and the regional comparison table.
PROFILE_INDICATORS: List[Tuple[str, str, str]] = [
    ("NY.GDP.PCAP.KD.ZG", "GDP per capita growth", "%"),
    ("EG.ELC.ACCS.ZS", "Electricity access", "% of population"),
    ("IT.NET.USER.ZS", "Internet use", "% of population"),
    ("NE.GDI.TOTL.ZS", "Capital formation", "% of GDP"),
    ("FP.CPI.TOTL.ZG", "Inflation", "annual %"),
    ("SP.DYN.LE00.IN", "Life expectancy", "years"),
]

# =============================================================================
# CACHED DATA LOADING
# =============================================================================

@st.cache_resource
def load_model() -> Tuple[SklearnPipeline, Dict]:
    """Load the trained model pipeline and metadata.

    Returns:
        Tuple of (fitted pipeline, metadata dict).
    """
    model_path = Path("models/growth_model.joblib")
    metadata_path = Path("models/model_metadata.json")

    if not model_path.exists():
        st.error(f"Model file not found: {model_path}")
        st.stop()
    if not metadata_path.exists():
        st.error(f"Metadata file not found: {metadata_path}")
        st.stop()

    pipeline = joblib.load(model_path)  # type: ignore[return-value]
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    logger.info("Loaded model: %s with %d features", metadata["model_type"], metadata["n_features"])
    return pipeline, metadata


@st.cache_data
def load_processed_data() -> pd.DataFrame:
    """Load the processed country-year panel data.

    Returns:
        DataFrame with iso3, country_name, year, features, and target.
    """
    data_path = Path("data/processed/model_data.parquet")
    if not data_path.exists():
        st.error(f"Processed data not found: {data_path}")
        st.stop()

    df = pd.read_parquet(data_path)
    logger.info("Loaded processed data: %s", df.shape)
    return df


@st.cache_data
def load_test_predictions() -> pd.DataFrame:
    """Load precomputed test set predictions.

    Returns:
        DataFrame with iso3, year, country_name, actual, predicted.
    """
    pred_path = Path("models/test_predictions.parquet")
    if not pred_path.exists():
        st.error(f"Test predictions not found: {pred_path}")
        st.stop()

    df = pd.read_parquet(pred_path)
    logger.info("Loaded test predictions: %s", df.shape)
    return df


@st.cache_data
def load_feature_importance() -> pd.DataFrame:
    """Load precomputed permutation importance (computed on VALIDATION with CIs).

    H1 remediation: the parquet now carries dispersion and significance columns;
    magnitudes carry no directional meaning and non-significant rows must be
    presented as 'not distinguishable from noise', never as effects.

    Returns:
        DataFrame with columns feature, importance_mean, importance_std,
        ci_lower, ci_upper, is_significant.
    """
    imp_path = Path("models/feature_importance.parquet")
    if not imp_path.exists():
        st.error(f"Feature importance not found: {imp_path}")
        st.stop()

    df = pd.read_parquet(imp_path)
    logger.info("Loaded feature importance for %d features (%d significant)",
                len(df), int(df["is_significant"].sum()))
    return df


@st.cache_data
def get_training_data(data: pd.DataFrame, train_end: int) -> pd.DataFrame:
    """Training-period rows only, for guardrail calibration.

    H3: spec section 14 requires observed TRAINING minimum/maximum. Using the
    full panel silently widens the safe band (inflation P99 92.05 vs 49.51).

    Args:
        data: Full processed country-year panel.
        train_end: Last training year from model metadata.

    Returns:
        Rows with year <= train_end.
    """
    return data[data["year"] <= train_end]


@st.cache_data
def load_country_metadata() -> pd.DataFrame:
    """Load country metadata (ISO3 codes and names)."""
    meta_path = Path("data/processed/country_metadata.csv")
    if not meta_path.exists():
        st.error(f"Country metadata not found: {meta_path}")
        st.stop()

    df = pd.read_csv(meta_path)
    logger.info("Loaded country metadata: %d countries", len(df))
    return df


@st.cache_data
def load_coordinates() -> pd.DataFrame:
    """Load the country centroid reference used by the Africa dot map.

    Returns:
        DataFrame with columns iso3, country_name, lat, lon. Empty frame when
        the reference file is missing, so the map degrades quietly instead of
        breaking the page.
    """
    try:
        return load_africa_coordinates()
    except FileNotFoundError as exc:
        logger.warning("Africa coordinates unavailable: %s", exc)
        return pd.DataFrame(columns=["iso3", "country_name", "lat", "lon"])


@st.cache_data
def load_geometry() -> Dict[str, Any]:
    """Load the committed simplified Africa geometry for the map.

    Returns:
        Geometry payload dict, or an empty dict when the reference file is
        missing so the page degrades quietly instead of breaking.
    """
    try:
        return load_africa_geometry()
    except FileNotFoundError as exc:
        logger.warning("Africa geometry unavailable: %s", exc)
        return {}


@st.cache_data
def compute_indicator_coverage(data: pd.DataFrame, feature_names: List[str]) -> Dict[str, float]:
    """Share of observed (non-missing) indicator values per country.

    Args:
        data: Full processed country-year panel.
        feature_names: Model input columns to score.

    Returns:
        Dict mapping iso3 to coverage in [0, 1].
    """
    cols = [f for f in feature_names if f in data.columns]
    if not cols:
        return {}
    coverage = data.groupby("iso3")[cols].apply(lambda g: 1.0 - g.isna().to_numpy().mean())
    return {str(k): float(v) for k, v in coverage.items()}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_feature_display_name(feature_code: str, config_features: List[Dict]) -> str:
    """Map WDI feature code to human-readable name.

    NY.GDP.PCAP.KD.ZG is carried as a predictor (this year's growth) but lives
    under `target` in the config rather than `features`, so the lookup below
    misses it and the raw code leaks into the UI. Named explicitly.
    """
    for feat in config_features:
        if feat["code"] == feature_code:
            return feat["name"]
    if feature_code == "NY.GDP.PCAP.KD.ZG":
        return "GDP per capita growth, current year (annual %)"
    return feature_code


def get_model_responsive_features(pipeline: SklearnPipeline, feature_names: List[str]) -> List[str]:
    """Feature codes the fitted model can actually respond to.

    B12: the deployed HistGradientBoosting model splits on only 8 of its 14
    inputs; the remaining 6 were never selected by any tree, so changing them
    cannot move a prediction by even a floating-point ulp. Reading the split
    features straight off the fitted predictors is exact: no threshold, no
    proxy metric. Falls back to all features if the estimator does not expose
    the private tree structure (e.g. a Ridge pipeline or a future sklearn).

    Args:
        pipeline: The fitted sklearn pipeline.
        feature_names: Ordered feature codes matching the model's input columns.

    Returns:
        Feature codes used in at least one split, ordered by split count
        (most-used first). All features if introspection is unavailable.
    """
    model = pipeline.steps[-1][1]
    predictors = getattr(model, "_predictors", None)
    if not predictors:
        return list(feature_names)

    split_counts: Dict[int, int] = {}
    try:
        for stage in predictors:
            for predictor in stage:
                for node in predictor.nodes:
                    if not node["is_leaf"]:
                        idx = int(node["feature_idx"])
                        split_counts[idx] = split_counts.get(idx, 0) + 1
    except (AttributeError, KeyError, IndexError, TypeError):
        return list(feature_names)

    if not split_counts:
        return list(feature_names)

    ordered = sorted(split_counts.items(), key=lambda kv: -kv[1])
    return [feature_names[i] for i, _ in ordered if i < len(feature_names)]


def probe_feature_responsiveness(
    pipeline: SklearnPipeline,
    baseline_row: pd.DataFrame,
    train_data: pd.DataFrame,
    feature_names: List[str],
    n_probe: int = 9,
) -> Dict[str, float]:
    """How much each feature can move the prediction *for this exact row*.

    Being split on somewhere in the forest is necessary but not sufficient: a
    tree-ensemble prediction only responds to a feature if this row actually
    reaches a node that splits on it. Ghana 2019, for example, is routed away
    from every internet-usage split, so that slider is inert for Ghana even
    though the model uses it elsewhere. The only exact test is to sweep the
    value and watch the prediction, which is what this does: one batched
    predict over all features and probe points.

    Args:
        pipeline: Fitted pipeline.
        baseline_row: Single-row frame of the country-year's feature values.
        train_data: Training-window rows, used for each feature's sweep range.
        feature_names: Ordered model input columns.
        n_probe: Probe points per feature across the training range.

    Returns:
        Feature code -> prediction spread (max - min) across its sweep.
    """
    probe_rows: List[pd.DataFrame] = []
    owners: List[str] = []
    for feat in feature_names:
        if feat not in train_data.columns:
            continue
        vals = train_data[feat].dropna()
        if len(vals) == 0:
            continue
        lo, hi = float(vals.min()), float(vals.max())
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            continue
        block = pd.concat([baseline_row] * n_probe, ignore_index=True)
        block[feat] = np.linspace(lo, hi, n_probe)
        probe_rows.append(block)
        owners.extend([feat] * n_probe)

    if not probe_rows:
        return {}

    probes = pd.concat(probe_rows, ignore_index=True).reindex(columns=feature_names)
    preds = pipeline.predict(probes)
    out: Dict[str, float] = {}
    owner_arr = np.asarray(owners)
    for feat in dict.fromkeys(owners):
        block_preds = preds[owner_arr == feat]
        out[feat] = float(block_preds.max() - block_preds.min())
    return out


def get_scenario_features(
    metadata: Dict,
    config_features: List[Dict],
    pipeline: Optional[SklearnPipeline] = None,
    responsiveness: Optional[Dict[str, float]] = None,
    tolerance: float = 1e-9,
) -> List[str]:
    """Select 3-5 features suitable for scenario exploration.

    B12 FIX: this used to return a hardcoded wish list of "policy-relevant"
    indicators (electricity, internet, capital formation, trade, inflation).
    The deployed model never splits on five of them, so every slider on the
    page was inert: the prediction could not change no matter how far a user
    dragged them. Selection is now driven by measured responsiveness for the
    country-year on screen (see probe_feature_responsiveness), falling back to
    the model's global split features, then to the raw feature list.
    Interpretable indicators are still preferred; actually moving the
    prediction is the hard constraint.
    """
    # Interpretable, policy-legible indicators, in preference order.
    priority_codes = [
        "EG.ELC.ACCS.ZS",      # Electricity access (%)
        "IT.NET.USER.ZS",      # Internet usage (%)
        "NE.GDI.TOTL.ZS",      # Gross capital formation (% of GDP)
        "NE.TRD.GNFS.ZS",      # Trade openness (% of GDP)
        "FP.CPI.TOTL.ZG",      # Inflation (%)
        "SP.DYN.LE00.IN",      # Life expectancy (years)
        "NY.GDP.PCAP.CD",      # GDP per capita (current US$)
        "SP.POP.GROW",         # Population growth (annual %)
        "BX.KLT.DINV.WD.GD.ZS",  # FDI net inflows (% of GDP)
        "SP.URB.TOTL.IN.ZS",   # Urban population (%)
        "FS.AST.PRVT.GD.ZS",   # Domestic credit to private sector (% of GDP)
    ]

    available_features = set(metadata["feature_names"])
    if responsiveness:
        # Measured for this row: keep only features that actually move it,
        # most responsive first.
        responsive = [
            f for f, spread in sorted(responsiveness.items(), key=lambda kv: -kv[1])
            if spread > tolerance
        ]
    elif pipeline is not None:
        responsive = get_model_responsive_features(pipeline, metadata["feature_names"])
    else:
        responsive = list(metadata["feature_names"])
    responsive_set = set(responsive)

    # Interpretable AND responsive, in preference order.
    scenario_features = [
        f for f in priority_codes if f in available_features and f in responsive_set
    ]

    # Top up from the remaining responsive features (most-used first).
    if len(scenario_features) < 5:
        for f in responsive:
            if f not in scenario_features:
                scenario_features.append(f)
            if len(scenario_features) >= 5:
                break

    # Last resort: if the model exposed no usable features at all, fall back to
    # the raw feature list so the page still renders.
    if len(scenario_features) < 3:
        for f in metadata["feature_names"]:
            if f not in scenario_features:
                scenario_features.append(f)
            if len(scenario_features) >= 3:
                break
    return scenario_features[:5]


def get_feature_range(data: pd.DataFrame, feature: str) -> Tuple[float, float]:
    """Get min and max of a feature from the processed data (full observed range)."""
    if feature in data.columns:
        vals = data[feature].dropna()
        if len(vals) > 0:
            return float(vals.min()), float(vals.max())
    return 0.0, 100.0


def get_feature_percentiles(data: pd.DataFrame, feature: str) -> Tuple[float, float]:
    """Get 1st and 99th percentiles for extrapolation warning."""
    if feature in data.columns:
        vals = data[feature].dropna()
        if len(vals) > 0:
            return float(vals.quantile(0.01)), float(vals.quantile(0.99))
    return 0.0, 100.0


def prepare_scenario_input(
    baseline_values: Dict[str, float],
    scenario_changes: Dict[str, float],
    feature_names: List[str],
) -> pd.DataFrame:
    """Create a single-row DataFrame for model prediction.

    Args:
        baseline_values: Dict of feature_name -> current value.
        scenario_changes: Dict of feature_name -> new value (subset of features).
        feature_names: Ordered list of feature names expected by model.

    Returns:
        Single-row DataFrame with columns in model's expected order.
    """
    # Start with baseline values
    row = {feat: baseline_values.get(feat, np.nan) for feat in feature_names}

    # Apply scenario changes
    for feat, new_val in scenario_changes.items():
        if feat in row:
            row[feat] = new_val

    df = pd.DataFrame([row])
    df = df.reindex(columns=feature_names)
    return df


def check_extrapolation_warning(
    scenario_values: Dict[str, float],
    data: pd.DataFrame,
) -> List[str]:
    """Check if any scenario values fall outside the TRAINING P1-P99 range.

    H3: callers must pass the training-window rows (see get_training_data),
    per spec section 14: guardrails are calibrated on observed training data.

    Args:
        scenario_values: Feature code -> proposed value.
        data: Training-period rows of the processed panel.

    Returns:
        List of warning messages.
    """
    warnings = []
    for feat, val in scenario_values.items():
        p1, p99 = get_feature_percentiles(data, feat)
        if val < p1 or val > p99:
            display_name = get_feature_display_name(feat, load_config_features())
            warnings.append(
                f"{display_name} = {val:.1f} is outside the training range "
                f"(P1 to P99: {p1:.1f} to {p99:.1f}). "
                f"The result may be unreliable due to extrapolation."
            )
    return warnings


@st.cache_data
def load_config_features() -> List[Dict]:
    """Load feature definitions from config for display names."""
    import yaml
    with open("config/indicators.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config["features"]


def render_causal_disclaimer(location: str = "overview") -> None:
    """Render the causal interpretation guardrail as an editorial research note.

    The substance is unchanged from the original disclaimer: the model
    describes statistical association, not the causal effect of intervening on
    a single indicator.

    Args:
        location: Where the disclaimer appears ('overview' or 'scenario').

    Returns:
        None.
    """
    if location == "overview":
        ui.render(ui.guardrail([
            "This application uses machine learning for prediction and "
            "decision support, not for causal policy-effect estimation.",
            "The model identifies statistical associations between "
            "development indicators and future GDP per capita growth. It "
            "cannot prove that changing one indicator will cause a particular "
            "change in growth.",
            "For example, if raising electricity access from 70% to 80% in "
            "the Scenario Explorer raises the predicted growth, the model is "
            "associating that feature profile with higher predicted growth. "
            "It is not evidence that raising electricity access alone would "
            "cause that increase.",
            "The distinction between prediction, association, causality and "
            "intervention effects is fundamental to responsible use of this "
            "tool.",
        ]))
    elif location == "scenario":
        ui.render(ui.guardrail([
            "Scenario results show how the predictive model responds to "
            "alternative indicator values. They are not causal estimates of "
            "the effect of implementing a specific policy.",
            "Use them as analytical evidence about the model, not as "
            "cause-and-effect claims about a country.",
        ]))


def safe_get_country_data(data: pd.DataFrame, iso3: str, year: int) -> Optional[pd.Series]:
    """Safely get country-year data, returning None if not found.

    Guards against IndexError when country has no growth data.
    """
    mask = (data["iso3"] == iso3) & (data["year"] == year)
    filtered = data[mask]
    if filtered.empty:
        return None
    return filtered.iloc[0]


def format_value(value: Any, spec: str = "{:.1f}", missing: str = "Not observed") -> str:
    """Format a possibly missing numeric value for display.

    Args:
        value: Number or NaN.
        spec: Format spec applied to present values.
        missing: Text used when the value is missing.

    Returns:
        Formatted string.
    """
    if value is None or (isinstance(value, float) and not np.isfinite(value)) or pd.isna(value):
        return missing
    return spec.format(value)


def split_summary(metadata: Dict) -> str:
    """One-line description of the temporal split, taken from metadata.

    Args:
        metadata: model_metadata.json contents.

    Returns:
        Plain sentence describing training, validation and test target years.
    """
    years = metadata.get("split_target_years", {})
    train = years.get("train", [None, None])
    val = years.get("val", [None, None])
    test = years.get("test", [None, None])
    return (
        f"Target years: training {train[0]} to {train[1]}, "
        f"validation {val[0]} to {val[1]}, test {test[0]} to {test[1]}."
    )


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main Streamlit application entry point."""

    inject_editorial_styles()

    # Load all cached resources
    pipeline, metadata = load_model()
    processed_data = load_processed_data()
    test_predictions = load_test_predictions()
    feature_importance = load_feature_importance()
    country_metadata = load_country_metadata()
    config_features = load_config_features()

    page, selected_iso3, selected_country_name = render_sidebar(
        country_metadata, metadata
    )

    if page == "Project Overview":
        render_overview_page(metadata, config_features, feature_importance,
                             processed_data)
    elif page == "Explore Africa":
        render_explore_page(processed_data, selected_iso3, selected_country_name,
                            config_features)
    elif page == "Model Performance":
        render_performance_page(test_predictions, feature_importance, metadata,
                                config_features)
    elif page == "Scenario Explorer":
        render_scenario_page(
            pipeline, processed_data, metadata, selected_iso3,
            selected_country_name, config_features
        )


def render_sidebar(country_metadata: pd.DataFrame, metadata: Dict) -> Tuple[str, str, str]:
    """Render the editorial navigation rail and shared country selector.

    Args:
        country_metadata: Frame with iso3 and country_name.
        metadata: model_metadata.json contents, used for the footer note.

    Returns:
        Tuple of (selected page label, selected iso3, selected country name).
    """
    with st.sidebar:
        ui.render(
            '<div class="ed-brand">'
            '<p class="ed-brand-top">Africa</p>'
            '<p class="ed-brand-main">Growth Explorer</p>'
            "</div>"
        )

        ui.render('<p class="ed-rail-label">Sections</p>')
        if "page" not in st.session_state:
            st.session_state["page"] = PAGES[0]
        for label in PAGES:
            active = st.session_state["page"] == label
            if st.button(
                label,
                key=f"nav_{label}",
                type="primary" if active else "tertiary",
                width="stretch",
            ):
                st.session_state["page"] = label
                st.rerun()
        page = st.session_state["page"]

        ui.render('<hr class="ed-rule" style="margin:20px 0 16px 0;" />')

        options = (
            country_metadata[["iso3", "country_name"]]
            .dropna()
            .sort_values("country_name")
        )
        names = options["country_name"].tolist()
        codes = options["iso3"].tolist()
        default_idx = names.index("Kenya") if "Kenya" in names else 0

        selected_idx = st.selectbox(
            "Country",
            range(len(codes)),
            index=default_idx,
            format_func=lambda i: names[i],
            key="country_selector",
        )
        selected_iso3 = codes[selected_idx]
        selected_country_name = names[selected_idx]

        ui.render(
            f'<p class="ed-meta" style="margin-top:6px;">ISO3 {ui.esc(selected_iso3)}. '
            "Used on Explore Africa and Scenario Explorer.</p>"
        )

        ui.render('<hr class="ed-rule" style="margin:24px 0 16px 0;" />')
        ui.render(
            '<p class="ed-meta">'
            f'Model: {ui.esc(metadata["model_type"])}<br />'
            f'Indicators: {metadata["n_features"]}<br />'
            f'Features observed to {metadata["train_end"]} in training<br />'
            f'{ui.esc(split_summary(metadata))}<br /><br />'
            "Predictive association tool. Not a causal inference engine."
            "</p>"
        )

    return page, selected_iso3, selected_country_name


# =============================================================================
# PAGE 1: PROJECT OVERVIEW
# =============================================================================

def render_overview_page(
    metadata: Dict,
    config_features: List[Dict],
    feature_importance: pd.DataFrame,
    processed_data: pd.DataFrame,
):
    """Render the Project Overview page.

    Args:
        metadata: model_metadata.json contents.
        config_features: Indicator definitions from config for display names.
        feature_importance: Validation permutation importance with CIs.
        processed_data: Full processed country-year panel.
    """
    coords = load_coordinates()
    geometry = load_geometry()
    coverage = compute_indicator_coverage(processed_data, metadata["feature_names"])
    test_metrics = metadata["metrics"]["winner_test"]
    sig = metadata.get("significance")
    n_countries = int(processed_data["iso3"].nunique())
    year_min = int(processed_data["year"].min())
    year_max = int(processed_data["year"].max())

    visual = ""
    if geometry.get("countries"):
        visual = ui.africa_map_svg(geometry["countries"], coverage)
    elif not coords.empty:
        visual = ui.africa_dot_svg(coverage_points(coords, coverage))

    ui.render(ui.hero(
        eyebrow="Africa Growth Explorer",
        title="Predicting near-term GDP per capita growth across African countries.",
        lead=(
            "A machine-learning decision-support system built from World Bank "
            "Development Indicators."
        ),
        sub=(
            f"Indicators observed in year t are used to estimate GDP per "
            f"capita growth in year t+1, for {n_countries} countries between "
            f"{year_min} and {year_max}."
        ),
        visual=visual,
    ))

    improvement = sig["paired_mae_improvement_vs_global_mean"] if sig else float("nan")
    ui.render(ui.kpi_grid([
        {
            "label": "Countries",
            "value": f"{n_countries}",
            "note": "African countries in the modelling panel",
            "tooltip": "Countries with at least one usable country-year row "
                       "after coverage filtering.",
        },
        {
            "label": "Indicators",
            "value": f"{metadata['n_features']}",
            "note": "World Bank indicators observed in year t",
            "tooltip": "Selected by at least 60% coverage on the training "
                       "window.",
        },
        {
            "label": "Test MAE",
            "value": f"{test_metrics['mae']:.2f} pp",
            "note": "Mean absolute error on held-out test data",
            "accent": True,
            "tooltip": "Mean absolute error in percentage points on target "
                       "years the model never saw.",
        },
        {
            "label": "Gain over mean baseline",
            "value": f"{improvement:+.2f} pp",
            "note": "Paired improvement whose 95% interval includes zero",
            "tooltip": "Bootstrap paired improvement in MAE against the "
                       "global-mean baseline on the test set.",
        },
    ]))

    # ------------------------------------------------------------------ how
    ui.render(ui.section(
        "How it works",
        "Five steps from published indicators to an explorable prediction.",
    ))
    ui.render(ui.process_strip([
        ("Collect", "World Bank development indicators for African countries."),
        ("Engineer", "Coverage filtering, lagged framing and log transforms."),
        ("Train", "Temporal split, no random shuffling, baseline gate."),
        ("Predict", "Next-year GDP per capita growth for each country-year."),
        ("Explore", "Scenarios, comparisons and error diagnostics."),
    ]))

    # ------------------------------------------------- indicator importance
    imp = feature_importance.copy()
    imp["label"] = imp["feature"].map(
        lambda f: get_feature_display_name(f, config_features)
    )
    n_sig = int(imp["is_significant"].sum())

    ui.render(ui.section(
        "Key indicators driving the prediction",
        "Permutation importance measured on the validation window. Terracotta "
        f"bars are the {n_sig} indicators whose interval excludes zero; the "
        "remainder are not distinguishable from noise.",
    ))
    chart_col, note_col = st.columns([3, 2], gap="large")
    with chart_col:
        series = pd.Series(imp["importance_mean"].to_numpy(), index=imp["label"].to_numpy())
        flags = pd.Series(imp["is_significant"].to_numpy(), index=imp["label"].to_numpy())
        fig = plot_feature_importance(
            series, title="", top_n=14, significant=flags,
        )
        st.pyplot(fig)
        plt.close(fig)
    with note_col:
        ui.render(ui.prose(
            "Importance here means how much validation error increases when a "
            "column is shuffled. It carries no direction: a high value does "
            "not say whether more of an indicator raises or lowers predicted "
            "growth."
        ))
        ui.render(ui.prose(
            f"Only {n_sig} of {len(imp)} indicators clear that bar. That is "
            "consistent with the model's overall parity with a mean baseline, "
            "and it is reported rather than smoothed over."
        ))
        ui.render(ui.legend([
            (COLORS["terracotta"], "Interval excludes zero"),
            (COLORS["inactive"], "Not distinguishable from noise"),
        ]))

    # ----------------------------------------------------------- coverage
    if geometry.get("countries"):
        ui.render(ui.section(
            "Data coverage",
            "Share of indicator values actually observed for each country "
            "across the panel, before median imputation.",
        ))
        map_col, text_col = st.columns([3, 2], gap="large")
        with map_col:
            fig = plot_africa_map(
                geometry, coverage, coordinates=coords, title="",
                value_label="Observed share of indicator values",
            )
            st.pyplot(fig)
            plt.close(fig)
        with text_col:
            observed = np.mean(list(coverage.values())) if coverage else float("nan")
            ui.render(ui.kpi_grid([
                {
                    "label": "Mean coverage",
                    "value": f"{observed:.0%}",
                    "note": "Average share of observed indicator values per country",
                },
                {
                    "label": "Panel rows",
                    "value": f"{len(processed_data):,}",
                    "note": f"Country-year observations, {year_min} to {year_max}",
                },
            ]))
            ui.render(ui.prose(
                "Missing values are filled with the training median inside the "
                "pipeline. Countries with thin coverage therefore lean on the "
                "panel median rather than on their own history, which is a "
                "reason to treat their predictions with more caution."
            ))
            ui.render(ui.prose(
                "Countries drawn in grey have no rows in the modelling panel."
            ))

    # -------------------------------------------------------- model note
    ui.render(ui.section("Model note"))
    if sig:
        verdict = (
            "distinguishable from predicting the mean"
            if sig["significant_at_95"]
            else "not distinguishable from predicting the mean"
        )
        ui.render(ui.callout(
            "Research note",
            [
                f"On the held-out test set the model reaches "
                f"{test_metrics['mae']:.2f} pp mean absolute error against "
                f"{metadata['metrics']['global_mean_baseline']['mae']:.2f} pp "
                "for the global-mean baseline.",
                f"The paired 95% interval on that improvement is "
                f"[{sig['ci_lower']:+.2f}, {sig['ci_upper']:+.2f}] pp and "
                f"includes zero, so the model is {verdict}. That parity result "
                "is the study's substantive finding, not a defect to be hidden.",
            ],
        ))

    # ----------------------------------------------------- technical detail
    with st.expander("View the indicator list and technical metadata"):
        feature_df = pd.DataFrame([
            {
                "Indicator": get_feature_display_name(feat, config_features),
                "Theme": next(
                    (f.get("theme", "") for f in config_features if f["code"] == feat),
                    "Target series",
                ),
                "Log transform": (
                    "Yes" if feat in metadata.get("log_transform_features", []) else "No"
                ),
                "WDI code": feat,
            }
            for feat in metadata["feature_names"]
        ])
        ui.render(ui.research_table(
            feature_df,
            note=(
                "Preprocessing: median imputation fitted on the training "
                "window; log1p applied inside the pipeline where marked."
            ),
        ))

    # ------------------------------------------------------- intended users
    ui.render(ui.section(
        "Intended users",
        "Built for analysts who need a fast, documented read on near-term "
        "growth conditions, with the model's limits stated up front.",
    ))
    users_col, use_col = st.columns(2, gap="large")
    with users_col:
        ui.render(ui.card("Who it is for", (
            '<p class="ed-prose">Development and economic analysts, policy '
            "analysts, government planning teams, development institutions, "
            "and researchers comparing development conditions across "
            "countries.</p>"
        )))
    with use_col:
        ui.render(ui.card("How to use it", (
            '<p class="ed-prose">For screening and analytical support. Not for '
            "final policy decisions. Combine model output with expert "
            "knowledge and country-specific evidence.</p>"
        ), soft=True))

    ui.render(ui.section("Causal guardrail"))
    render_causal_disclaimer("overview")


# =============================================================================
# PAGE 2: EXPLORE AFRICA
# =============================================================================

def render_explore_page(
    data: pd.DataFrame,
    selected_iso3: str,
    selected_country_name: str,
    config_features: List[Dict],
):
    """Render the Explore Africa page with country trends and comparisons.

    Args:
        data: Full processed country-year panel.
        selected_iso3: Selected country ISO3 code.
        selected_country_name: Selected country display name.
        config_features: Indicator definitions for display names.
    """
    country_data = data[data["iso3"] == selected_iso3].copy()

    ui.render(ui.page_header(
        "Explore Africa",
        selected_country_name,
        "Development indicators over time, and how this country compares with "
        "the rest of the panel.",
    ))

    if country_data.empty:
        ui.render(ui.callout(
            "No data",
            [f"No rows are available for {selected_country_name} "
             f"({selected_iso3})."],
            kind="critical",
        ))
        return

    growth_col = "NY.GDP.PCAP.KD.ZG"
    target_col = "target_next_year"
    latest_year = int(country_data["year"].max())
    latest_row = country_data[country_data["year"] == latest_year].iloc[0]

    ui.render(ui.section(
        "Observed growth",
        f"GDP per capita growth for {selected_country_name}, with the "
        "next-year series the model is trained to predict.",
    ))
    fig = plot_growth_trend(
        years=country_data["year"].tolist(),
        observed=country_data[growth_col].tolist(),
        next_year=country_data[target_col].tolist(),
        title="",
    )
    st.pyplot(fig)
    plt.close(fig)

    # ------------------------------------------------------ current profile
    ui.render(ui.section(
        "Current profile",
        f"Latest observed values, {latest_year}.",
    ))
    ui.render(ui.kpi_grid([
        {
            "label": label,
            "value": format_value(latest_row.get(code, np.nan)),
            "note": unit,
            "accent": code == growth_col,
            "tooltip": get_feature_display_name(code, config_features),
        }
        for code, label, unit in PROFILE_INDICATORS
    ]))

    # ----------------------------------------------------- indicator trends
    ui.render(ui.section(
        "Indicator trends",
        "Each indicator is drawn on its own scale, so levels stay readable.",
    ))
    feature_cols = [
        c for c in data.columns
        if c not in ["iso3", "country_name", "year", "target_next_year"]
    ]
    default_features = [code for code, _, _ in PROFILE_INDICATORS if code in feature_cols][:6]
    selected_features = st.multiselect(
        "Indicators",
        options=feature_cols,
        default=default_features,
        format_func=lambda x: get_feature_display_name(x, config_features),
        key="explore_features",
    )

    if selected_features:
        display_names = {
            f: get_feature_display_name(f, config_features) for f in selected_features
        }
        fig = plot_indicator_small_multiples(country_data, selected_features, display_names)
        st.pyplot(fig)
        plt.close(fig)
    else:
        ui.render(ui.prose("Select one or more indicators to draw their trends."))

    # -------------------------------------------------- regional comparison
    latest_year_all = int(data["year"].max())
    latest_all = data[data["year"] == latest_year_all].copy()

    ui.render(ui.section(
        "Regional comparison",
        f"All countries in the panel, {latest_year_all}. The selected country "
        "is highlighted.",
    ))

    compare_cols = ["country_name"] + [
        code for code, _, _ in PROFILE_INDICATORS if code in latest_all.columns
    ]
    comparison = latest_all[compare_cols].sort_values("country_name").reset_index(drop=True)
    comparison.columns = ["Country"] + [
        f"{label} ({unit})" for code, label, unit in PROFILE_INDICATORS
        if code in latest_all.columns
    ]
    highlight_rows = comparison.index[
        comparison["Country"] == selected_country_name
    ].tolist()

    ui.render(ui.research_table(
        comparison,
        formats={c: "{:.1f}" for c in comparison.columns if c != "Country"},
        highlight_rows=highlight_rows,
        scroll=True,
        note=(
            "Values are the latest published observations in the panel. Blank "
            "cells are missing in the source data and are imputed with the "
            "training median before the model sees them."
        ),
    ))


# =============================================================================
# PAGE 3: MODEL PERFORMANCE
# =============================================================================

def render_performance_page(
    test_predictions: pd.DataFrame,
    feature_importance: pd.DataFrame,
    metadata: Dict,
    config_features: List[Dict],
):
    """Render the Model Performance page as a research appendix.

    Args:
        test_predictions: Deployed model's frozen test predictions.
        feature_importance: Validation permutation importance with CIs.
        metadata: model_metadata.json contents (single source of truth).
        config_features: Indicator definitions for display names.
    """
    metrics = metadata["metrics"]
    test_metrics = metrics["winner_test"]
    sig = metadata.get("significance")
    winner_mae = test_metrics["mae"]
    gm_mae = metrics["global_mean_baseline"]["mae"]

    ui.render(ui.page_header(
        "Model performance",
        "What the model gets right, and what it does not",
        "How the deployed model behaves on held-out data, how it compares "
        "with simple baselines, and where its errors are systematic.",
    ))

    # ------------------------------------------------------------- verdict
    ui.render(ui.section("Model verdict"))
    if sig:
        significant = sig["significant_at_95"]
        headline = (
            "The model is statistically distinguishable from the global-mean "
            "baseline on the held-out test set."
            if significant else
            "The model is not statistically distinguishable from the "
            "global-mean baseline on the held-out test set."
        )
        ui.render(ui.callout(
            "Model verdict",
            [
                headline,
                f"Test MAE is {winner_mae:.2f} pp against {gm_mae:.2f} pp for "
                f"the global mean. The paired 95% interval on the improvement "
                f"is [{sig['ci_lower']:+.2f}, {sig['ci_upper']:+.2f}] pp, from "
                f"{sig['n_bootstrap']:,} bootstrap resamples.",
                "This parity result is the study's substantive finding. The "
                "protocol behind it was fixed before the test set was touched; "
                "see the report for the pre-registered procedure.",
            ],
            kind="warning" if not significant else "info",
        ))

    gate = metadata.get("gate", {})
    if gate:
        gate_text = (
            "Selection gate on validation: passed. The model beat every "
            "validation baseline before artifacts were written."
            if gate.get("passed") else
            "Selection gate on validation: failed. The model shipped via "
            "--allow-baseline-failure and this is disclosed in the report."
        )
        ui.render(ui.meta(gate_text))

    ui.render(ui.kpi_grid([
        {
            "label": "Test MAE",
            "value": f"{winner_mae:.2f} pp",
            "note": f"Global-mean baseline: {gm_mae:.2f} pp",
            "accent": True,
            "tooltip": "Mean absolute error in percentage points on the "
                       "held-out test set.",
        },
        {
            "label": "Test RMSE",
            "value": f"{test_metrics['rmse']:.2f} pp",
            "note": "Root mean squared error, penalises large misses",
            "tooltip": "Root mean squared error on the held-out test set.",
        },
        {
            "label": "Test R squared",
            "value": f"{test_metrics['r2']:.3f}",
            "note": "Share of test variance explained",
            "tooltip": "Coefficient of determination on the held-out test set.",
        },
        {
            "label": "Directional skill",
            "value": f"{test_metrics['directional_skill']:+.1%}",
            "note": (
                f"Accuracy {test_metrics['directional_accuracy']:.1%} against a "
                f"majority-class rate of "
                f"{test_metrics['directional_majority_rate']:.1%}"
            ),
            "tooltip": "Directional accuracy minus the majority-class rate. "
                       "Values near zero mean no directional information "
                       "beyond the class prior.",
        },
    ]))

    # -------------------------------------------------- baseline comparison
    ui.render(ui.section(
        "Baseline comparison",
        "The same test set scored by the deployed model and by three "
        "reference predictors.",
    ))

    def _row(label: str, m: Dict) -> Dict:
        return {
            "Predictor": label,
            "MAE": m.get("mae"),
            "RMSE": m.get("rmse"),
            "R squared": m.get("r2"),
            "Directional accuracy": m.get("directional_accuracy"),
            "Majority-class rate": m.get("directional_majority_rate"),
            "Directional skill": m.get("directional_skill"),
        }

    rows = [
        _row("Global mean baseline", metrics["global_mean_baseline"]),
        _row("Persistence baseline", metrics.get("persistence_baseline", {})),
    ]
    if "country_historical_mean_baseline" in metrics:
        rows.append(_row("Country historical mean baseline",
                         metrics["country_historical_mean_baseline"]))
    rows.append(_row(f"{metadata['model_type']} (deployed)", test_metrics))
    baseline_df = pd.DataFrame(rows)

    ui.render(ui.research_table(
        baseline_df,
        formats={
            "MAE": "{:.3f}",
            "RMSE": "{:.3f}",
            "R squared": "{:.3f}",
            "Directional accuracy": "{:.1%}",
            "Majority-class rate": "{:.1%}",
            "Directional skill": "{:+.1%}",
        },
        emphasis=["Predictor"],
        highlight_rows=[len(rows) - 1],
        note=(
            "Raw directional accuracy equals the majority-class rate for any "
            "constant-sign predictor, so it is reported next to that rate. "
            "Directional skill is accuracy minus the majority rate."
        ),
    ))

    # ---------------------------------------------------- error diagnostics
    ui.render(ui.section(
        "Actual against predicted",
        "Points on the dashed line are exact. The vertical spread of the "
        "predictions shows how little the model varies its estimate.",
    ))
    scatter_col, resid_col = st.columns(2, gap="large")
    with scatter_col:
        fig = plot_actual_vs_predicted(
            actual=test_predictions["actual"].values,
            predicted=test_predictions["predicted"].values,
            title="Test set",
        )
        st.pyplot(fig)
        plt.close(fig)
    with resid_col:
        fig = plot_residuals(
            actual=test_predictions["actual"].values,
            predicted=test_predictions["predicted"].values,
            title="Residuals",
        )
        st.pyplot(fig)
        plt.close(fig)

    ui.render(ui.prose(
        "The residual cloud is centred on zero with no strong trend against "
        "the predicted value, so the error is close to random rather than "
        "systematically biased. The narrow range of predictions is the "
        "substantive limitation: the model rarely departs far from the mean."
    ))

    # -------------------------------------------------- feature importance
    imp = feature_importance.copy()
    imp["label"] = imp["feature"].map(
        lambda f: get_feature_display_name(f, config_features)
    )
    significant_rows = imp[imp["is_significant"]]
    noise_rows = imp[~imp["is_significant"]]

    ui.render(ui.section(
        "Feature importance",
        f"Permutation importance on the validation set. "
        f"{len(significant_rows)} of {len(imp)} indicators have an interval "
        "that excludes zero.",
    ))

    if len(significant_rows) > 0:
        series = pd.Series(imp["importance_mean"].to_numpy(), index=imp["label"].to_numpy())
        flags = pd.Series(imp["is_significant"].to_numpy(), index=imp["label"].to_numpy())
        fig = plot_feature_importance(series, title="", top_n=14, significant=flags)
        st.pyplot(fig)
        plt.close(fig)
        ui.render(ui.legend([
            (COLORS["terracotta"], "Interval excludes zero"),
            (COLORS["inactive"], "Not distinguishable from noise"),
        ]))
    else:
        ui.render(ui.callout(
            "Attribution",
            ["No feature has a permutation-importance interval excluding "
             "zero. On this model, feature-level attribution is "
             "indistinguishable from noise, and no directional claims are "
             "made."],
            kind="warning",
        ))

    ui.render(ui.meta(
        "Importance magnitude carries no directional meaning. Sign claims come "
        "only from the linear benchmark's standardized coefficients, below."
    ))

    if len(noise_rows) > 0:
        with st.expander(
            f"View indicators not distinguishable from noise ({len(noise_rows)})"
        ):
            noise_display = noise_rows[
                ["label", "importance_mean", "ci_lower", "ci_upper", "feature"]
            ].rename(columns={
                "label": "Indicator",
                "importance_mean": "Importance",
                "ci_lower": "Interval lower",
                "ci_upper": "Interval upper",
                "feature": "WDI code",
            })
            ui.render(ui.research_table(
                noise_display,
                formats={
                    "Importance": "{:+.4f}",
                    "Interval lower": "{:+.4f}",
                    "Interval upper": "{:+.4f}",
                },
            ))

    with st.expander("View ridge standardized coefficients (direction view)"):
        rc_path = Path("models/ridge_coefficients.parquet")
        if rc_path.exists():
            rc = pd.read_parquet(rc_path)
            rc = rc.assign(
                Indicator=rc["feature"].map(
                    lambda f: get_feature_display_name(
                        f.replace("_log1p", ""), config_features
                    )
                )
            )
            rc_display = rc[["Indicator", "coefficient", "feature"]].rename(
                columns={"coefficient": "Standardized coefficient",
                         "feature": "Model column"}
            )
            ui.render(ui.research_table(
                rc_display,
                formats={"Standardized coefficient": "{:+.4f}"},
                semantic=["Standardized coefficient"],
                note=(
                    "Association only, from the linear benchmark fitted on the "
                    "training window. Coefficients use the post-transform "
                    "column order. Not causal."
                ),
            ))
        else:
            ui.render(ui.prose(
                "Run scripts/finalize_model.py to generate "
                "ridge_coefficients.parquet."
            ))

    # ------------------------------------------------------ yearly metrics
    ui.render(ui.section(
        "Performance by year",
        "Error by reference year, against the global-mean baseline. Each row "
        "predicts growth in the following year.",
    ))
    yearly_metrics = test_predictions.groupby("year").apply(
        lambda g: pd.Series({
            "MAE": np.mean(np.abs(g["actual"] - g["predicted"])),
            "RMSE": np.sqrt(np.mean((g["actual"] - g["predicted"])**2)),
            "Directional accuracy": ((g["actual"] >= 0) == (g["predicted"] >= 0)).mean(),
            "Majority rate": max((g["actual"] >= 0).mean(), (g["actual"] < 0).mean()),
            "Observations": len(g),
        }), include_groups=False
    ).reset_index().rename(columns={"year": "Reference year"})

    chart_col, table_col = st.columns([2, 3], gap="large")
    with chart_col:
        fig = plot_metric_by_year(
            years=yearly_metrics["Reference year"].tolist(),
            values=yearly_metrics["MAE"].tolist(),
            reference=gm_mae,
            title="",
            reference_label="Global mean",
        )
        st.pyplot(fig)
        plt.close(fig)
    with table_col:
        ui.render(ui.research_table(
            yearly_metrics,
            formats={
                "Reference year": "{:.0f}",
                "MAE": "{:.3f}",
                "RMSE": "{:.3f}",
                "Directional accuracy": "{:.1%}",
                "Majority rate": "{:.1%}",
                "Observations": "{:.0f}",
            },
        ))

    # ---------------------------------------------------------- limitations
    ui.render(ui.section(
        "Limitations",
        "Read these alongside every number on this page.",
    ))
    ui.render(ui.numbered_notes([
        (
            "Parity with the baseline",
            "The paired 95% interval against the global-mean baseline "
            "includes zero. Treat predictions as the mean plus noise rather "
            "than as country-specific signal.",
        ),
        (
            "Temporal generalization only",
            "Evaluation covers future years of countries seen during "
            "training. Nothing here tests performance on a country the model "
            "has never seen.",
        ),
        (
            "Association is not causation",
            "Feature importance and scenario responses describe predictive "
            "association, not causal effect. See the causal guardrail on the "
            "Overview and Scenario Explorer pages.",
        ),
        (
            "COVID-19 period",
            "Validation target years cover the 2020 shock; test target years "
            f"are post-COVID. The refit policy is pre-registered as "
            f"'{metadata.get('refit_strategy', 'train_only')}'.",
        ),
        (
            "Limited feature set",
            f"{metadata['n_features']} WDI indicators. Conflict, commodity "
            "prices, governance and weather are among the growth determinants "
            "left out.",
        ),
        (
            "Median imputation",
            "Missing values are filled with the training median, which may "
            "not reflect country-specific conditions, particularly where "
            "coverage is thin.",
        ),
    ]))


# =============================================================================
# PAGE 4: SCENARIO EXPLORER
# =============================================================================

def render_scenario_page(
    pipeline: SklearnPipeline,
    processed_data: pd.DataFrame,
    metadata: Dict,
    selected_iso3: str,
    selected_country_name: str,
    config_features: List[Dict],
):
    """Render the Scenario Explorer page.

    Args:
        pipeline: Fitted model pipeline.
        processed_data: Full processed country-year panel.
        metadata: model_metadata.json contents.
        selected_iso3: Selected country ISO3 code.
        selected_country_name: Selected country display name.
        config_features: Indicator definitions for display names.
    """
    ui.render(ui.page_header(
        "Scenario Explorer",
        selected_country_name,
        "Change observed indicator values and watch how the deployed model "
        "revises its prediction for the following year.",
    ))

    years_series = processed_data[processed_data["iso3"] == selected_iso3]["year"].dropna()
    country_years = sorted(years_series.unique().tolist())

    if len(country_years) == 0:
        ui.render(ui.callout(
            "No data",
            [f"No rows are available for {selected_country_name}."],
            kind="critical",
        ))
        return

    year_col, _ = st.columns([1, 3])
    with year_col:
        selected_year = st.selectbox(
            "Reference year",
            options=country_years,
            index=len(country_years) - 1,
            key="scenario_year",
            help="Indicators are read from this year; the model predicts "
                 "growth in the following year.",
        )

    baseline_row = safe_get_country_data(processed_data, selected_iso3, int(selected_year))
    if baseline_row is None:
        ui.render(ui.callout(
            "No data",
            [f"No row for {selected_country_name} in {selected_year}."],
            kind="critical",
        ))
        return

    feature_names = metadata["feature_names"]
    baseline_values = {feat: baseline_row.get(feat, np.nan) for feat in feature_names}
    missing_features = [f for f, v in baseline_values.items() if pd.isna(v)]

    train_data = get_training_data(processed_data, metadata["train_end"])

    # ------------------------------------------------------ current profile
    ui.render(ui.section(
        "Current profile",
        f"Observed indicator values for {selected_country_name} in "
        f"{int(selected_year)}.",
    ))
    profile_codes = [c for c, _, _ in PROFILE_INDICATORS if c in feature_names]
    ui.render(ui.kpi_grid([
        {
            "label": label,
            "value": format_value(baseline_values.get(code, np.nan)),
            "note": unit,
            "tooltip": get_feature_display_name(code, config_features),
        }
        for code, label, unit in PROFILE_INDICATORS if code in profile_codes
    ]))

    if missing_features:
        ui.render(ui.callout(
            "Imputed inputs",
            [
                f"{len(missing_features)} of {len(feature_names)} indicators "
                f"are missing for {selected_country_name} in "
                f"{int(selected_year)}. The pipeline fills them with the "
                "training median before predicting.",
                ", ".join(
                    get_feature_display_name(f, config_features)
                    for f in missing_features
                ),
            ],
            kind="warning",
        ))

    with st.expander("View all observed feature values"):
        baseline_display = pd.DataFrame([
            {
                "Indicator": get_feature_display_name(f, config_features),
                "Observed value": float(v) if pd.notna(v) else np.nan,
                "WDI code": f,
            }
            for f, v in baseline_values.items()
        ])
        ui.render(ui.research_table(
            baseline_display,
            formats={"Observed value": "{:.2f}"},
            note="Blank values are missing in the source data and are imputed "
                 "with the training median.",
        ))

    # ---------------------------------------------------------- the controls
    # B12: only offer sliders the model can respond to *for this row*. The
    # deployed model splits on 8 of its 14 inputs, and for any given country
    # -year fewer than that are reachable, so a fixed list produces sliders
    # that silently do nothing.
    probe_baseline = pd.DataFrame(
        [{feat: baseline_values.get(feat, np.nan) for feat in feature_names}],
        columns=feature_names,
    )
    responsiveness = probe_feature_responsiveness(
        pipeline, probe_baseline, train_data, feature_names
    )
    scenario_features = get_scenario_features(
        metadata, config_features, pipeline, responsiveness
    )
    inert_features = [
        f for f in feature_names
        if responsiveness.get(f, 0.0) <= 1e-9 and f not in scenario_features
    ]

    ui.render(ui.section(
        "Adjust the scenario",
        "Slider ranges and the P1 to P99 warning band come from the observed "
        f"training distribution, 2000 to {metadata['train_end']}.",
    ))

    if inert_features:
        ui.render(ui.callout(
            "Why these indicators",
            [
                "The controls below are limited to indicators that change the "
                f"model's prediction for {selected_country_name} "
                f"{int(selected_year)}. Sweeping the others across their full "
                "training range moves the prediction by nothing at all, "
                "because this country-year never reaches a decision point "
                "that uses them.",
            ],
        ))

    n_cols = 3
    cols = st.columns(n_cols, gap="large")
    scenario_changes: Dict[str, float] = {}
    clamp_notes: List[str] = []

    for i, feat in enumerate(scenario_features):
        with cols[i % n_cols]:
            display_name = get_feature_display_name(feat, config_features)
            data_min, data_max = get_feature_range(train_data, feat)
            p1, p99 = get_feature_percentiles(train_data, feat)
            baseline_val = baseline_values.get(feat, np.nan)

            if pd.notna(baseline_val):
                default_val = float(baseline_val)
            else:
                default_val = (p1 + p99) / 2

            # B9 (plan v3): clamp observed defaults into the training band so a
            # slider can never silently start outside guardrail support.
            clamped = float(np.clip(default_val, data_min, data_max))
            if abs(clamped - default_val) > 1e-9:
                clamp_notes.append(
                    f"{display_name}: observed {default_val:.1f} is outside "
                    f"the training range [{data_min:.1f}, {data_max:.1f}], so "
                    "the slider starts clamped."
                )
            default_val = clamped

            new_val = st.slider(
                display_name,
                min_value=float(data_min),
                max_value=float(data_max),
                value=float(default_val),
                step=(data_max - data_min) / 100,
                help=f"Observed value: "
                     f"{format_value(baseline_val, '{:.1f}', 'missing')}. "
                     f"Training range {data_min:.1f} to {data_max:.1f}, "
                     f"P1 to P99 {p1:.1f} to {p99:.1f}.",
                # Key is scoped to the country-year: without this, switching
                # country keeps the previous country's slider positions while
                # the Baseline column updates, so the two silently disagree.
                key=f"slider_{selected_iso3}_{int(selected_year)}_{feat}",
            )
            observed_note = (
                f"Observed {baseline_val:.1f}" if pd.notna(baseline_val)
                else "No observed value; starts at the training midpoint"
            )
            ui.render(ui.meta(
                f"{observed_note} | training P1 to P99 {p1:.1f} to {p99:.1f}"
            ))
            scenario_changes[feat] = new_val

    if clamp_notes:
        ui.render(ui.callout("Clamped controls", clamp_notes, kind="warning"))

    extrapolation = check_extrapolation_warning(scenario_changes, train_data)
    if extrapolation:
        ui.render(ui.callout("Extrapolation", extrapolation, kind="warning"))

    if inert_features:
        with st.expander(
            f"View excluded indicators ({len(inert_features)} cannot move this "
            "prediction)"
        ):
            ui.render(ui.prose(
                f"For {selected_country_name} {int(selected_year)}, sweeping "
                "each of these across its full training range leaves the "
                "prediction unchanged. Rather than show sliders that silently "
                "do nothing, the page leaves them out. The set differs by "
                "country and year."
            ))
            excluded_df = pd.DataFrame([
                {
                    "Indicator": get_feature_display_name(f, config_features),
                    "Prediction spread (pp)": responsiveness.get(f, 0.0),
                    "WDI code": f,
                }
                for f in inert_features
            ])
            ui.render(ui.research_table(
                excluded_df,
                formats={"Prediction spread (pp)": "{:.3f}"},
                note=(
                    "This matches the evaluation: shuffling most of these "
                    "columns at random leaves the model's error unchanged."
                ),
            ))

    # --------------------------------------------------------- the prediction
    scenario_input = prepare_scenario_input(baseline_values, scenario_changes, feature_names)
    baseline_input = pd.DataFrame(
        [{feat: baseline_values.get(feat, np.nan) for feat in feature_names}],
        columns=feature_names,
    )
    baseline_pred = pipeline.predict(baseline_input)[0]
    scenario_pred = pipeline.predict(scenario_input)[0]
    difference = scenario_pred - baseline_pred

    ui.render(ui.section(
        "Prediction",
        f"Predicted GDP per capita growth for {int(selected_year) + 1}.",
    ))
    ui.render(ui.kpi_grid([
        {
            "label": "Baseline",
            "value": f"{baseline_pred:.2f}%",
            "note": "Observed indicator values, with median imputation where "
                    "they are missing",
            "tooltip": "Model prediction using the observed profile for this "
                       "country-year.",
        },
        {
            "label": "Scenario",
            "value": f"{scenario_pred:.2f}%",
            "note": "Your adjusted indicator values",
            "accent": True,
            "tooltip": "Model prediction after applying the slider values.",
        },
        {
            "label": "Change",
            "value": f"{difference:+.2f} pp",
            "note": "Scenario minus baseline, in percentage points",
            "tone": ("positive" if difference > 0
                     else "negative" if difference < 0 else None),
            "tooltip": "Difference between the two model outputs. Not a "
                       "causal effect estimate.",
        },
    ]))

    # ----------------------------------------------- what moved the prediction
    # H1: multiplying a permutation importance by a raw unit change is
    # dimensionally meaningless, so that heuristic is gone; each row below
    # re-runs the deployed model changing one indicator at a time.
    ui.render(ui.section(
        "What moved the prediction",
        "Each row re-runs the model changing only that indicator.",
    ))
    change_contributions = []
    for feat, new_val in scenario_changes.items():
        baseline_val = baseline_values.get(feat, np.nan)
        if pd.isna(baseline_val):
            continue
        probe = baseline_input.copy()
        probe[feat] = new_val
        delta = float(pipeline.predict(probe)[0] - baseline_pred)
        # B13: keep these numeric. Pre-formatting them as strings made the
        # sort below call abs() on str and raise TypeError.
        change_contributions.append({
            "Indicator": get_feature_display_name(feat, config_features),
            "Baseline": float(baseline_val),
            "Scenario": float(new_val),
            "Change": float(new_val) - float(baseline_val),
            "Model response (pp)": delta,
        })

    if change_contributions:
        contrib_df = pd.DataFrame(change_contributions)
        contrib_df = contrib_df.sort_values(
            "Model response (pp)",
            key=lambda s: s.abs(),
            ascending=False,
        ).reset_index(drop=True)

        max_abs = float(contrib_df["Model response (pp)"].abs().max())
        display_df = contrib_df.copy()
        display_df["Response"] = [
            ui.bar_cell(v, max_abs) for v in contrib_df["Model response (pp)"]
        ]
        ui.render(ui.research_table(
            display_df,
            formats={
                "Baseline": "{:.1f}",
                "Scenario": "{:.1f}",
                "Change": "{:+.1f}",
                "Model response (pp)": "{:+.3f}",
            },
            emphasis=["Model response (pp)"],
            semantic=["Model response (pp)"],
            raw_html=["Response"],
            note=(
                "Individual responses need not sum to the total because the "
                "model is non-linear. These are the model's responses to "
                "different inputs, not causal effects."
            ),
        ))

        if max_abs < 5e-4:
            ui.render(ui.callout(
                "No movement",
                ["None of the indicators you moved changed the prediction "
                 "meaningfully. That is a real property of this model, not a "
                 "fault in the page: it barely responds to these inputs, "
                 "which is the same finding the evaluation reports."],
                kind="warning",
            ))

    ui.render(ui.section("Causal guardrail"))
    render_causal_disclaimer("scenario")

    ui.render(ui.meta(
        f"Model: {metadata['model_type']} | Indicators: "
        f"{metadata['n_features']} | Training window: 2000 to "
        f"{metadata['train_end']} | {split_summary(metadata)}"
    ))


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
