"""Africa Growth Explorer - Streamlit Application.

Decision-support system for predicting near-term GDP per capita growth
across African countries using World Bank Development Indicators.

This application loads a pre-trained model and provides:
- Project Overview with causal disclaimer
- Explore Africa: Country-level indicator trends and comparisons
- Model Performance: Metrics, actual vs predicted, feature importance
- Scenario Explorer: Interactive what-if predictions with extrapolation warnings
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Import project visualization functions
from src.visualization import (
    set_project_style,
    get_project_palette,
    plot_actual_vs_predicted,
    plot_residuals,
    plot_feature_importance,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Apply project visual style (must be before any matplotlib usage)
set_project_style()
PALETTE = get_project_palette()

# Set matplotlib backend for Streamlit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Africa Growth Explorer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Type alias for sklearn pipeline
from sklearn.pipeline import Pipeline as SklearnPipeline

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
    inputs — the remaining 6 were never selected by any tree, so changing them
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
    value and watch the prediction, which is what this does — one batched
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
    page was inert — the prediction could not change no matter how far a user
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
    per spec section 14 — guardrails are calibrated on observed training data.

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
                f"**Warning:** {display_name} = {val:.1f} is outside the training "
                f"range (P1-P99: {p1:.1f}–{p99:.1f}). "
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
    """Render the causal interpretation disclaimer.

    Args:
        location: Where the disclaimer appears ('overview' or 'scenario').
    """
    if location == "overview":
        st.markdown("---")
        st.warning(
            "⚠️ **Causal Interpretation Disclaimer**\n\n"
            "This application uses machine learning for **prediction and decision support**, "
            "not causal policy-effect estimation.\n\n"
            "The model identifies statistical associations between development indicators "
            "and future GDP per capita growth. It **cannot prove** that changing one "
            "indicator will cause a particular change in growth.\n\n"
            "For example: If increasing electricity access from 70% to 80% in the Scenario "
            "Explorer leads to a higher predicted growth, this means the model *associates* "
            "that feature profile with higher predicted growth. It does **not** prove that "
            "increasing electricity access alone will cause the predicted increase.\n\n"
            "This distinction between **prediction**, **association**, **causality**, and "
            "**intervention effects** is fundamental to responsible use of this tool."
        )
    elif location == "scenario":
        st.error(
            "⚠️ **Important:** Scenario results show how the predictive model responds "
            "to alternative indicator values. They should **not** be interpreted as "
            "causal estimates of the effect of implementing a specific policy."
        )


def safe_get_country_data(data: pd.DataFrame, iso3: str, year: int) -> Optional[pd.Series]:
    """Safely get country-year data, returning None if not found.

    Guards against IndexError when country has no growth data.
    """
    mask = (data["iso3"] == iso3) & (data["year"] == year)
    filtered = data[mask]
    if filtered.empty:
        return None
    return filtered.iloc[0]


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main Streamlit application entry point."""

    # Load all cached resources
    pipeline, metadata = load_model()
    processed_data = load_processed_data()
    test_predictions = load_test_predictions()
    feature_importance = load_feature_importance()
    country_metadata = load_country_metadata()
    config_features = load_config_features()

    # Sidebar: Country selector (shared across pages)
    st.sidebar.title("🌍 Africa Growth Explorer")
    st.sidebar.markdown("---")

    country_options = country_metadata.set_index("iso3")["country_name"].to_dict()
    country_codes = sorted(country_options.keys())
    country_labels = [f"{code} - {country_options[code]}" for code in country_codes]

    selected_idx = st.sidebar.selectbox(
        "Select Country",
        range(len(country_codes)),
        format_func=lambda i: country_labels[i],
        key="country_selector",
    )
    selected_iso3 = country_codes[selected_idx]
    selected_country_name = country_options[selected_iso3]

    st.sidebar.markdown(f"**Selected:** {selected_country_name} ({selected_iso3})")
    st.sidebar.markdown("---")

    # Navigation
    page = st.sidebar.radio(
        "Navigation",
        ["📋 Project Overview", "🌍 Explore Africa", "📊 Model Performance", "🎯 Scenario Explorer"],
        key="page_selector",
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        f"Model: {metadata['model_type']}  \n"
        f"Features: {metadata['n_features']}  \n"
        f"Train features ≤{metadata['train_end']} (targets ≤{metadata['train_end'] + 1})  \n"
        f"Test features >{metadata['val_end']} (targets >{metadata['val_end'] + 1})"
    )

    # Route to page
    if page == "📋 Project Overview":
        render_overview_page(metadata, config_features)
    elif page == "🌍 Explore Africa":
        render_explore_page(processed_data, selected_iso3, selected_country_name, config_features)
    elif page == "📊 Model Performance":
        render_performance_page(test_predictions, feature_importance, metadata, processed_data)
    elif page == "🎯 Scenario Explorer":
        render_scenario_page(
            pipeline, processed_data, metadata, selected_iso3, selected_country_name,
            config_features
        )


# =============================================================================
# PAGE 1: PROJECT OVERVIEW
# =============================================================================

def render_overview_page(metadata: Dict, config_features: List[Dict]):
    """Render the Project Overview page."""
    st.title("📋 Project Overview")
    st.markdown("### Africa Growth Explorer: ML Decision-Support System")

    st.markdown("""
    **Core Question:** To what extent can recent development indicators predict near-term
    GDP per capita growth across African countries, and which observed development
    conditions are most informative for those predictions?

    **Decision-Support Question:** Given a country's current development profile, what
    level of next-year GDP per capita growth does the model estimate, which indicators
    contribute most to that estimate, and how does the estimate change under alternative
    development scenarios?
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎯 Problem Statement")
        st.markdown("""
        - **Target:** GDP per capita growth (annual %) in year *t+1*
        - **Features:** Development indicators observed in year *t*
        - **Geography:** African countries from an explicit ISO3 list (see README for coverage notes)
        - **Time Range:** 2000–2024 (training: 2000–2017, validation: 2018–2020, test: 2021+)
        - **Task:** Regression (predict next-year growth in percentage points)
        """)

        st.subheader("📊 Data Source")
        st.markdown("""
        - **World Bank World Development Indicators (WDI)**
        - 14 candidate indicators across 6 themes
        - Source: [datatopics.worldbank.org/world-development-indicators](https://datatopics.worldbank.org/world-development-indicators/)
        """)

    with col2:
        st.subheader("🤖 Model")
        st.markdown(f"""
        - **Algorithm:** {metadata['model_type']}
        - **Features:** {metadata['n_features']} (selected by ≥60% coverage on training data)
        - **Preprocessing:** Median imputation (fitted on training data)
        - **Validation:** Temporal split (no random splitting to avoid leakage)
        - **Baselines:** Global mean, Persistence (current year's growth)
        """)

        st.subheader("📈 Key Metrics (Test Set)")
        test_metrics = metadata["metrics"]["winner_test"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("MAE", f"{test_metrics['mae']:.2f} pp",
                  help=f"Global-mean baseline: {metadata['metrics']['global_mean_baseline']['mae']:.2f} pp")
        m2.metric("RMSE", f"{test_metrics['rmse']:.2f} pp")
        m3.metric("R²", f"{test_metrics['r2']:.3f}")
        m4.metric("Directional accuracy",
                  f"{test_metrics['directional_accuracy']:.1%}",
                  help=f"Majority-class rate is "
                       f"{test_metrics['directional_majority_rate']:.1%}; skill = "
                       f"{test_metrics['directional_skill']:+.1%} (H4: raw accuracy "
                       f"is not skill)")
        sig_ov = metadata.get("significance")
        if sig_ov:
            st.caption(
                f"Paired improvement vs global mean: "
                f"{sig_ov['paired_mae_improvement_vs_global_mean']:+.2f} pp, "
                f"95% CI [{sig_ov['ci_lower']:+.2f}, {sig_ov['ci_upper']:+.2f}]"
                + (" — significant." if sig_ov["significant_at_95"]
                   else " — spans zero: parity with the mean baseline.")
            )

    st.subheader("📋 Features Used by Model")
    feature_df = pd.DataFrame([
        {
            "Code": feat,
            "Name": get_feature_display_name(feat, config_features),
            "Log Transform": "✓" if feat in metadata.get("log_transform_features", []) else "",
        }
        for feat in metadata["feature_names"]
    ])
    st.dataframe(feature_df, use_container_width=True, hide_index=True)

    st.subheader("🎯 Intended Users")
    st.markdown("""
    - Development analysts
    - Economic researchers
    - Policy analysts
    - Government planning teams
    - NGOs and development institutions
    - Students and researchers comparing development conditions
    """)

    st.info(
        "💡 **This tool is designed for screening and analytical support.** "
        "It is **not intended to make final policy decisions**. "
        "Always combine model output with expert knowledge and country-specific evidence."
    )

    # Causal disclaimer on Overview page
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
    """Render the Explore Africa page with country trends and comparisons."""
    st.title("🌍 Explore Africa")
    st.markdown(f"### {selected_country_name} ({selected_iso3}) - Development Indicators Over Time")

    # Filter data for selected country
    country_data = data[data["iso3"] == selected_iso3].copy()

    if country_data.empty:
        st.error(f"No data available for {selected_country_name} ({selected_iso3})")
        return

    # Growth trend chart
    st.subheader("📈 GDP Per Capita Growth Trend")
    growth_col = "NY.GDP.PCAP.KD.ZG"
    target_col = "target_next_year"

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        country_data["year"],
        country_data[growth_col],
        marker="o",
        color=PALETTE["secondary"],
        label="Observed Growth (t)",
        linewidth=2,
    )
    ax.plot(
        country_data["year"],
        country_data[target_col],
        marker="s",
        color=PALETTE["accent"],
        label="Next-Year Growth (t+1, Target)",
        linewidth=2,
        linestyle="--",
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("GDP Per Capita Growth (%)")
    ax.set_title(f"{selected_country_name} - Growth Trend")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

    # Latest available data summary
    st.subheader("📋 Latest Available Data")
    latest_year = int(country_data["year"].max())  # type: ignore[arg-type]
    latest_row = country_data[country_data["year"] == latest_year].iloc[0]  # type: ignore[return-value]

    # Display key indicators in columns
    key_indicators = [
        ("NY.GDP.PCAP.KD.ZG", "GDP Growth (%)"),
        ("EG.ELC.ACCS.ZS", "Electricity Access (%)"),
        ("IT.NET.USER.ZS", "Internet Usage (%)"),
        ("NE.GDI.TOTL.ZS", "Capital Formation (% GDP)"),
        ("FP.CPI.TOTL.ZG", "Inflation (%)"),
        ("SP.DYN.LE00.IN", "Life Expectancy (years)"),
    ]

    cols = st.columns(3)
    for i, (code, label) in enumerate(key_indicators):
        with cols[i % 3]:
            val = latest_row.get(code, np.nan)
            if pd.notna(val):
                st.metric(label, f"{val:.1f}")
            else:
                st.metric(label, "N/A")

    # Indicator trends (multi-select)
    st.subheader("📊 Indicator Trends")
    feature_cols = [c for c in data.columns if c not in ["iso3", "country_name", "year", "target_next_year"]]

    # Group features by theme for easier selection
    features_by_theme = {}
    for feat in config_features:
        if feat["code"] in feature_cols:
            theme = feat.get("theme", "Other")
            if theme not in features_by_theme:
                features_by_theme[theme] = []
            features_by_theme[theme].append(feat["code"])

    selected_features = st.multiselect(
        "Select indicators to plot",
        options=feature_cols,
        default=[k for v in features_by_theme.values() for k in v[:2]],
        format_func=lambda x: get_feature_display_name(x, config_features),
        key="explore_features",
    )

    if selected_features:
        fig, ax = plt.subplots(figsize=(10, 5))
        for feat in selected_features:
            vals = country_data[feat].dropna()
            if len(vals) > 0:
                ax.plot(
                    country_data.loc[vals.index, "year"],
                    vals,
                    marker="o",
                    label=get_feature_display_name(feat, config_features),
                    linewidth=2,
                )
        ax.set_xlabel("Year")
        ax.set_ylabel("Value")
        ax.set_title(f"{selected_country_name} - Selected Indicators")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # Country comparison table (latest year)
    st.subheader("🌍 Regional Comparison (Latest Available Year)")
    latest_year_all = int(data["year"].max())  # type: ignore[arg-type]
    latest_all = data[data["year"] == latest_year_all].copy()

    compare_cols = ["iso3", "country_name"] + [c for c, _ in key_indicators]
    compare_cols = [c for c in compare_cols if c in latest_all.columns]

    comparison = latest_all[compare_cols].sort_values("country_name")  # type: ignore[arg-type]
    comparison_display = comparison.copy()
    comparison_display.columns = ["ISO3", "Country"] + [lbl for _, lbl in key_indicators if _ in comparison.columns]

    st.dataframe(comparison_display, use_container_width=True, hide_index=True)


# =============================================================================
# PAGE 3: MODEL PERFORMANCE
# =============================================================================

def render_performance_page(
    test_predictions: pd.DataFrame,
    feature_importance: pd.DataFrame,
    metadata: Dict,
    processed_data: pd.DataFrame,
):
    """Render the Model Performance page.

    Args:
        test_predictions: Deployed model's frozen test predictions.
        feature_importance: Validation permutation importance with CIs.
        metadata: model_metadata.json contents (single source of truth).
        processed_data: Full processed panel (for by-context lookups).
    """
    st.title("📊 Model Performance")

    metrics = metadata["metrics"]

    # ------------------------------------------------------------------
    # Honest headline (Task 2.4/3.3): the significance verdict, not spin.
    # ------------------------------------------------------------------
    sig = metadata.get("significance")
    winner_mae = metrics["winner_test"]["mae"]
    gm_mae = metrics["global_mean_baseline"]["mae"]
    if sig:
        verdict = (
            "statistically distinguishable from predicting the mean"
            if sig["significant_at_95"] else
            "**not** statistically distinguishable from predicting the mean"
        )
        st.info(
            f"Test MAE {winner_mae:.2f} vs {gm_mae:.2f} for the global-mean baseline. "
            f"Paired 95% CI on the improvement: "
            f"[{sig['ci_lower']:+.2f}, {sig['ci_upper']:+.2f}] pp — this includes zero: "
            f"the model is {verdict}. This parity result is the study's substantive "
            f"finding; see the report for the pre-registered protocol behind it."
        )

    gate = metadata.get("gate", {})
    if gate:
        st.caption(
            "Selection gate (validation): "
            + ("PASSED — model beats all validation baselines before artifacts were written"
               if gate.get("passed") else
               "FAILED — shipped via --allow-baseline-failure; disclose this in the report")
        )

    # Baseline comparison
    st.subheader("🏁 Baseline Comparison (Test Set)")

    def _row(label: str, m: Dict) -> Dict:
        return {
            "Model": label,
            "MAE": m.get("mae"),
            "RMSE": m.get("rmse"),
            "R²": m.get("r2"),
            "Directional accuracy": m.get("directional_accuracy"),
            "Majority-class rate": m.get("directional_majority_rate"),
            "Directional skill": (
                m["directional_skill"] if m.get("directional_skill") is not None else None
            ),
        }

    rows = [
        _row("Global Mean Baseline", metrics["global_mean_baseline"]),
        _row("Persistence Baseline", metrics.get("persistence_baseline", {})),
    ]
    if "country_historical_mean_baseline" in metrics:
        rows.append(_row("Country Historical Mean Baseline",
                         metrics["country_historical_mean_baseline"]))
    rows.append(_row(f"{metadata['model_type']} (Test)", metrics["winner_test"]))
    baseline_df = pd.DataFrame(rows)

    st.dataframe(
        baseline_df.style.format({
            "MAE": "{:.3f}",
            "RMSE": "{:.3f}",
            "R²": "{:.3f}",
            "Directional accuracy": "{:.1%}",
            "Majority-class rate": "{:.1%}",
            "Directional skill": "{:+.1%}",
        }),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "H4: raw directional accuracy equals the majority-class rate for any "
        "constant-sign predictor. 'Directional skill' = accuracy − majority rate; "
        "values near zero mean no directional information beyond the class prior."
    )

    # Actual vs Predicted
    st.subheader("🎯 Actual vs. Predicted (Test Set)")
    fig = plot_actual_vs_predicted(
        actual=test_predictions["actual"].values,
        predicted=test_predictions["predicted"].values,
        title="Test Set: Actual vs. Predicted GDP Growth",
    )
    st.pyplot(fig)
    plt.close(fig)

    # Residuals
    st.subheader("📉 Residual Analysis")
    fig = plot_residuals(
        actual=test_predictions["actual"].values,
        predicted=test_predictions["predicted"].values,
        title="Test Set: Residuals vs. Predicted",
    )
    st.pyplot(fig)
    plt.close(fig)

    # Feature Importance (validation, with significance flags - H1)
    st.subheader("🔍 Feature Importance (Permutation, Validation Set)")
    imp = feature_importance
    significant = imp[imp["is_significant"]]
    noise = imp[~imp["is_significant"]]

    if len(significant) > 0:
        plot_series = pd.Series(
            significant["importance_mean"].values, index=significant["feature"].values)
        fig = plot_feature_importance(
            importance=plot_series,
            title="Permutation importance — CI excludes 0 (validation set)",
            top_n=14,
        )
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.warning(
            "No feature has a permutation-importance CI excluding zero: on this "
            "model, feature-level attribution is indistinguishable from noise. "
            "No directional claims are made."
        )
    st.caption(
        f"{len(significant)}/{len(imp)} features significant at 95%. Importance "
        "magnitude carries NO directional meaning (H1) — sign claims come only "
        "from Ridge coefficients, in the panel below."
    )
    if len(noise) > 0:
        with st.expander(f"Features not distinguishable from noise ({len(noise)})"):
            st.dataframe(
                noise[["feature", "importance_mean", "ci_lower", "ci_upper"]].style.format(
                    {"importance_mean": "{:+.4f}", "ci_lower": "{:+.4f}",
                     "ci_upper": "{:+.4f}"}),
                use_container_width=True, hide_index=True,
            )

    with st.expander("Ridge standardized coefficients (direction view, training fit)"):
        rc_path = Path("models/ridge_coefficients.parquet")
        if rc_path.exists():
            rc = pd.read_parquet(rc_path)
            st.dataframe(
                rc[["feature", "coefficient"]].style.format({"coefficient": "{:+.4f}"}),
                use_container_width=True, hide_index=True,
            )
            st.caption(
                "Association only, from the linear benchmark — coefficients use the "
                "post-ColumnTransformer feature order (H2-safe mapping). Not causal."
            )
        else:
            st.info("Run scripts/finalize_model.py to generate ridge_coefficients.parquet.")

    # Metrics by year
    st.subheader("📅 Performance by Year (Test Set)")
    yearly_metrics = test_predictions.groupby("year").apply(
        lambda g: pd.Series({
            "MAE": np.mean(np.abs(g["actual"] - g["predicted"])),
            "RMSE": np.sqrt(np.mean((g["actual"] - g["predicted"])**2)),
            "Directional Accuracy": ((g["actual"] >= 0) == (g["predicted"] >= 0)).mean(),
            "Majority Rate": max((g["actual"] >= 0).mean(), (g["actual"] < 0).mean()),
            "N": len(g),
        }), include_groups=False
    ).reset_index()

    st.dataframe(
        yearly_metrics.style.format({
            "MAE": "{:.3f}",
            "RMSE": "{:.3f}",
            "Directional Accuracy": "{:.1%}",
            "Majority Rate": "{:.1%}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    # Model limitations
    st.subheader("⚠️ Model Limitations")
    st.markdown(f"""
    - **Parity, not victory:** the model's paired 95% CI vs the global-mean baseline
      includes zero (see banner above). Treat predictions as the mean plus noise.
    - **Temporal generalization only:** Model evaluates prediction for future years of
      countries seen during training, not for completely unseen countries.
    - **Association ≠ Causation:** Feature importance reflects predictive association,
      not causal effect. See Causal Disclaimer on Overview and Scenario pages.
    - **COVID-19 period:** validation TARGET years (2019–2021) include the 2020 shock.
      Test TARGET years (2022–2024) are post-COVID. Refit policy is pre-registered
      (`{metadata.get('refit_strategy', 'train_only')}`).
    - **Limited features:** {metadata['n_features']} WDI indicators; many growth
      determinants omitted.
    - **Median imputation:** Missing values filled with training median, which may
      not reflect country-specific conditions.
    """)


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
    """Render the Scenario Explorer page."""
    st.title("🎯 Scenario Explorer")
    st.markdown(f"### {selected_country_name} ({selected_iso3}) - What-If Analysis")

    # Year selector
    years_series = processed_data[processed_data["iso3"] == selected_iso3]["year"].dropna()
    country_years = sorted(years_series.unique().tolist())

    if len(country_years) == 0:
        st.error(f"No data available for {selected_country_name}")
        return

    # Default to latest year with data
    default_year_idx = len(country_years) - 1
    selected_year = st.selectbox(
        "Reference Year (features observed in this year, predicts next year)",
        options=country_years,
        index=default_year_idx,
        key="scenario_year",
    )

    # Get baseline feature values for selected country-year
    baseline_row = safe_get_country_data(processed_data, selected_iso3, int(selected_year))  # type: ignore[arg-type]

    if baseline_row is None:
        st.error(f"No data for {selected_country_name} in {selected_year}")
        return

    feature_names = metadata["feature_names"]
    baseline_values = {feat: baseline_row.get(feat, np.nan) for feat in feature_names}

    # Identify missing baseline values
    missing_features = [f for f, v in baseline_values.items() if pd.isna(v)]
    if missing_features:
        st.warning(
            f"⚠️ {len(missing_features)} feature(s) missing for {selected_country_name} in {selected_year}. "
            f"Model will use training median imputation for: "
            f"{', '.join([get_feature_display_name(f, config_features) for f in missing_features])}"
        )

    # Show baseline values
    with st.expander("📋 Baseline Feature Values (Observed)", expanded=False):
        baseline_display = pd.DataFrame([
            {
                "Feature": get_feature_display_name(f, config_features),
                "Code": f,
                "Value": f"{v:.2f}" if pd.notna(v) else "Missing (will impute)",
            }
            for f, v in baseline_values.items()
        ])
        st.dataframe(baseline_display, use_container_width=True, hide_index=True)

    # Scenario controls. H3 FIX: all ranges/percentiles come from the TRAINING
    # window (spec section 14), not the full panel — otherwise a user could set
    # inflation to 80% and get no warning because 80% < full-panel P99 (92%).
    train_data = get_training_data(processed_data, metadata["train_end"])
    st.subheader("🎚️ Adjust Scenario Indicators")
    st.caption(
        "Modify up to 5 indicators. Slider ranges and P1–P99 warning bands are "
        f"the observed **training** distribution (2000–{metadata['train_end']}); "
        "values outside the training P1–P99 trigger extrapolation warnings."
    )

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
    if inert_features:
        with st.expander(
            f"Why these indicators? ({len(inert_features)} cannot move this "
            f"prediction)",
            expanded=False,
        ):
            st.markdown(
                f"For **{selected_country_name} {int(selected_year)}**, sweeping the "
                "following indicators across their full training range does not "
                "change the model's prediction at all — this country-year never "
                "reaches a decision point that uses them. Rather than show sliders "
                "that silently do nothing, the page leaves them out:\n\n"
                + "\n".join(
                    f"- {get_feature_display_name(f, config_features)}"
                    for f in inert_features
                )
                + "\n\nThis matches the evaluation: shuffling most of these columns "
                "at random leaves the model's error unchanged. Only 2 of 14 "
                "indicators had an effect distinguishable from noise. The set can "
                "differ by country and year."
            )

    scenario_changes = {}

    # Create sliders in columns
    n_cols = 3
    cols = st.columns(n_cols)

    for i, feat in enumerate(scenario_features):
        with cols[i % n_cols]:
            display_name = get_feature_display_name(feat, config_features)
            data_min, data_max = get_feature_range(train_data, feat)
            p1, p99 = get_feature_percentiles(train_data, feat)
            baseline_val = baseline_values.get(feat, np.nan)

            # Default to baseline if available, else midpoint of training P1-P99
            if pd.notna(baseline_val):
                default_val = float(baseline_val)
            else:
                default_val = (p1 + p99) / 2

            # B9 (plan v3): clamp observed defaults into the training band so a
            # slider can never silently start outside guardrail support.
            clamped = float(np.clip(default_val, data_min, data_max))
            if abs(clamped - default_val) > 1e-9:
                st.caption(
                    f"Observed value {default_val:.1f} is outside the training "
                    f"range [{data_min:.1f}, {data_max:.1f}]; slider clamped."
                )
            default_val = clamped

            # Slider with full observed training range
            new_val = st.slider(
                f"{display_name}",
                min_value=float(data_min),
                max_value=float(data_max),
                value=float(default_val),
                step=(data_max - data_min) / 100,
                help=f"Training range: {data_min:.1f}–{data_max:.1f} | "
                     f"training P1–P99: {p1:.1f}–{p99:.1f}",
                # Key is scoped to the country-year: without this, switching
                # country keeps the previous country's slider positions while
                # the Baseline column updates, so the two silently disagree.
                key=f"slider_{selected_iso3}_{int(selected_year)}_{feat}",
            )
            scenario_changes[feat] = new_val

    # Display warnings (single source of truth, training-calibrated)
    for w in check_extrapolation_warning(scenario_changes, train_data):
        st.warning(w)

    # Prepare input and predict
    scenario_input = prepare_scenario_input(baseline_values, scenario_changes, feature_names)

    # Baseline prediction (using original baseline values, imputed)
    baseline_input = pd.DataFrame([{feat: baseline_values.get(feat, np.nan) for feat in feature_names}], columns=feature_names)
    baseline_pred = pipeline.predict(baseline_input)[0]

    # Scenario prediction
    scenario_pred = pipeline.predict(scenario_input)[0]
    difference = scenario_pred - baseline_pred

    # Display predictions
    st.subheader("📊 Prediction Results")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Baseline Prediction",
            f"{baseline_pred:.2f}%",
            help="Model prediction using observed feature values (with median imputation for missing)"
        )
    with col2:
        st.metric(
            "Scenario Prediction",
            f"{scenario_pred:.2f}%",
            help="Model prediction with your adjusted indicator values"
        )
    with col3:
        st.metric(
            "Difference",
            f"{difference:+.2f} pp",
            delta=f"{difference:+.2f}",
            help="Scenario minus baseline (percentage points)"
        )

    # Show what actually moved the prediction: a TRUE one-at-a-time model delta.
    # H1: multiplying a permutation importance by a raw unit change is
    # dimensionally meaningless, so that heuristic ("Approx. Contribution")
    # is removed entirely; each row now re-runs the deployed model.
    st.subheader("🔬 Indicators Driving the Change")
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
            "Feature": get_feature_display_name(feat, config_features),
            "Baseline": float(baseline_val),
            "Scenario": float(new_val),
            "Change": float(new_val) - float(baseline_val),
            "Individual effect (pp)": delta,
        })

    if change_contributions:
        contrib_df = pd.DataFrame(change_contributions)
        contrib_df = contrib_df.sort_values(
            "Individual effect (pp)",
            key=lambda s: s.abs(),
            ascending=False,
        )
        st.dataframe(
            contrib_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Baseline": st.column_config.NumberColumn(format="%.1f"),
                "Scenario": st.column_config.NumberColumn(format="%.1f"),
                "Change": st.column_config.NumberColumn(format="%+.1f"),
                "Individual effect (pp)": st.column_config.NumberColumn(format="%+.3f"),
            },
        )
        if contrib_df["Individual effect (pp)"].abs().max() < 5e-4:
            st.info(
                "None of the indicators you moved changed the prediction "
                "meaningfully. That is a real property of this model, not a bug "
                "in the page — it barely responds to these inputs, which is the "
                "same finding the evaluation reports."
            )
        st.caption(
            "Each row re-runs the model changing only that indicator. Individual "
            "effects need not sum to the total because the model is non-linear. "
            "These are the model's responses to different inputs, not causal effects."
        )

    # Causal disclaimer on Scenario page
    render_causal_disclaimer("scenario")

    st.markdown("---")
    st.caption(
        "Model: " + metadata["model_type"] + " | "
        "Features: " + str(metadata["n_features"]) + " | "
        "Training data: 2000–2017 | "
        "This is a predictive association tool, not a causal inference engine."
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Import matplotlib here to avoid issues with Streamlit's threading
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    main()