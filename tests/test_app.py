"""Tests for app.py pure helpers (H5 remediation).

Covers the serving-side contracts the review flagged as untested:
scenario-input column ordering, NaN preservation for the imputer, training-
window guardrails (H3), extrapolation warnings, display fallbacks, and an
end-to-end prediction through the committed artifacts.

The Streamlit runtime is not started; module-level code executes in Streamlit
"bare mode". Guarded by importorskip so the suite passes on a test-only env.
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("streamlit")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

app = importlib.import_module("app")


# ----------------------------------------------------------------------------
# Scenario input contract
# ----------------------------------------------------------------------------

def test_prepare_scenario_input_column_order_matches_contract():
    """Model contract is positional+named: output order must equal feature_names."""
    feats = ["b_code", "a_code", "c_code"]
    baseline = {"a_code": 1.0, "b_code": 2.0, "c_code": 3.0}
    changes = {"a_code": 9.9}
    out = app.prepare_scenario_input(baseline, changes, feats)
    assert list(out.columns) == feats
    assert len(out) == 1
    assert out["a_code"].iloc[0] == 9.9            # scenario applied
    assert out["b_code"].iloc[0] == 2.0            # baseline preserved


def test_prepare_scenario_input_preserves_nan_for_imputer():
    """Missing features must arrive as NaN (in-pipeline median imputation)."""
    feats = ["x", "y"]
    out = app.prepare_scenario_input({"x": 1.0}, {}, feats)
    assert np.isnan(out["y"].iloc[0])


def test_prepare_scenario_input_ignores_unknown_scenario_keys():
    feats = ["x"]
    out = app.prepare_scenario_input({"x": 1.0}, {"not_a_feature": 5.0}, feats)
    assert list(out.columns) == ["x"]


# ----------------------------------------------------------------------------
# Guardrails: training-window calibration (H3)
# ----------------------------------------------------------------------------

@pytest.fixture()
def panel_with_regime_split():
    """Inflation moderate in train (max 40), extreme in test years (92)."""
    years = list(range(2000, 2025))
    rows = []
    for y in years:
        val = 30.0 if y <= 2017 else (5.0 if y <= 2020 else 92.0)
        rows.append({"iso3": "GHA", "country_name": "Ghana", "year": y,
                     "FP.CPI.TOTL.ZG": val})
    return pd.DataFrame(rows)


def test_get_training_data_filters_years(panel_with_regime_split):
    train = app.get_training_data(panel_with_regime_split, 2017)
    assert train["year"].max() <= 2017
    assert len(train) == 18


def test_get_feature_percentiles_uses_training_window_only(panel_with_regime_split):
    """H3: full-panel P99 for inflation is 92.05-style; training-only is 30."""
    train = app.get_training_data(panel_with_regime_split, 2017)
    p1_full, p99_full = app.get_feature_percentiles(panel_with_regime_split,
                                                     "FP.CPI.TOTL.ZG")
    p1_tr, p99_tr = app.get_feature_percentiles(train, "FP.CPI.TOTL.ZG")
    assert p99_full == pytest.approx(92.0)     # the WRONG (permissive) band
    assert p99_tr == pytest.approx(30.0)       # the guardrail we must use


def test_check_extrapolation_warning_fires_outside_training_band(panel_with_regime_split):
    train = app.get_training_data(panel_with_regime_split, 2017)
    warns = app.check_extrapolation_warning({"FP.CPI.TOTL.ZG": 80.0}, train)
    assert len(warns) == 1 and "training" in warns[0]
    ok = app.check_extrapolation_warning({"FP.CPI.TOTL.ZG": 30.0}, train)
    assert ok == []


def test_get_feature_range_training_window(panel_with_regime_split):
    train = app.get_training_data(panel_with_regime_split, 2017)
    lo, hi = app.get_feature_range(train, "FP.CPI.TOTL.ZG")
    assert (lo, hi) == (30.0, 30.0)
    lo_full, hi_full = app.get_feature_range(panel_with_regime_split, "FP.CPI.TOTL.ZG")
    assert hi_full == 92.0 and hi < hi_full   # training band strictly tighter


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------

def test_get_scenario_features_returns_three_to_five():
    meta = {"feature_names": ["EG.ELC.ACCS.ZS", "IT.NET.USER.ZS",
                              "NE.GDI.TOTL.ZS", "NE.TRD.GNFS.ZS",
                              "FP.CPI.TOTL.ZG", "SP.DYN.LE00.IN",
                              "NY.GDP.PCAP.CD"]}
    out = app.get_scenario_features(meta, [])
    assert 3 <= len(out) <= 5
    assert set(out) <= set(meta["feature_names"])


def test_get_scenario_features_pads_short_lists():
    """Fewer than 3 priority features: fall back to other available columns."""
    meta = {"feature_names": ["EG.ELC.ACCS.ZS", "XX.OTHER.A", "YY.OTHER.B",
                              "ZZ.OTHER.C"]}
    out = app.get_scenario_features(meta, [])
    assert len(out) >= 3


def test_get_feature_display_name_falls_back_to_code():
    feats = [{"code": "A.B.C", "name": "Alpha"}]
    assert app.get_feature_display_name("A.B.C", feats) == "Alpha"
    assert app.get_feature_display_name("Q.R.S", feats) == "Q.R.S"


def test_safe_get_country_data_returns_none_when_missing():
    df = pd.DataFrame({"iso3": ["GHA"], "year": [2019], "v": [1.0]})
    assert app.safe_get_country_data(df, "GHA", 2019) is not None
    assert app.safe_get_country_data(df, "KEN", 2019) is None
    assert app.safe_get_country_data(df, "GHA", 1999) is None


# ----------------------------------------------------------------------------
# End-to-end through committed artifacts (no retraining)
# ----------------------------------------------------------------------------

def test_end_to_end_prediction_from_committed_artifacts():
    """Deployed pipeline + metadata contract serve a finite prediction."""
    import joblib
    import json
    meta = json.loads((REPO_ROOT / "models/model_metadata.json").read_text())
    pipe = joblib.load(REPO_ROOT / "models/growth_model.joblib")
    panel = pd.read_parquet(REPO_ROOT / "data/processed/model_data.parquet")
    feats = meta["feature_names"]
    row = panel[(panel["iso3"] == "GHA") & (panel["year"] == 2019)].iloc[0]
    baseline = {f: float(row[f]) if pd.notna(row.get(f, np.nan)) else np.nan
                for f in feats}
    X = app.prepare_scenario_input(baseline, {}, feats)
    pred = float(pipe.predict(X)[0])
    assert np.isfinite(pred)
    # and the scenario probe still respects the contract
    X2 = app.prepare_scenario_input(baseline, {feats[0]: 500.0}, feats)
    assert np.isfinite(float(pipe.predict(X2)[0]))
    assert list(X2.columns) == feats


def test_load_feature_importance_returns_ci_frame():
    """App-facing parquet carries significance flags (H1 display contract)."""
    df = pd.read_parquet(REPO_ROOT / "models/feature_importance.parquet")
    assert {"is_significant", "ci_lower", "ci_upper"} <= set(df.columns)
    # the app must not choke on the new schema
    out = app.load_feature_importance.__wrapped__()
    assert isinstance(out, pd.DataFrame) and len(out) > 0


# ----------------------------------------------------------------------------
# B12: scenario sliders must be able to move the prediction
# ----------------------------------------------------------------------------

def _load_artifacts():
    import joblib
    import json
    meta = json.loads((REPO_ROOT / "models/model_metadata.json").read_text())
    pipe = joblib.load(REPO_ROOT / "models/growth_model.joblib")
    panel = pd.read_parquet(REPO_ROOT / "data/processed/model_data.parquet")
    return meta, pipe, panel


def test_deployed_model_ignores_some_features():
    """Documents the fact that motivates B12.

    If a future model does respond to every input this assertion fails loudly,
    which is the signal to revisit the slider-selection logic rather than a bug.
    """
    meta, pipe, _ = _load_artifacts()
    feats = meta["feature_names"]
    used = app.get_model_responsive_features(pipe, feats)
    assert set(used) < set(feats), (
        "Deployed model now splits on every feature; revisit get_scenario_features"
    )


def test_scenario_sliders_are_never_inert():
    """B12 regression: every offered slider must move the prediction.

    The original bug shipped five sliders — electricity, internet, capital
    formation, trade, inflation — that the model never splits on, so dragging
    them changed nothing. Checked across several countries because reachability
    is row-dependent.
    """
    meta, pipe, panel = _load_artifacts()
    feats = meta["feature_names"]
    train = app.get_training_data(panel, meta["train_end"])

    for iso3 in ["GHA", "NGA", "KEN", "ZAF", "EGY"]:
        sub = panel[panel["iso3"] == iso3]
        if sub.empty:
            continue
        year = int(sub["year"].max())
        row = sub[sub["year"] == year].iloc[0]
        baseline = {f: row.get(f, np.nan) for f in feats}
        probe_base = pd.DataFrame([baseline], columns=feats)

        resp = app.probe_feature_responsiveness(pipe, probe_base, train, feats)
        selected = app.get_scenario_features(meta, [], pipe, resp)
        assert 3 <= len(selected) <= 5

        for feat in selected:
            vals = train[feat].dropna()
            lo, hi = float(vals.min()), float(vals.max())
            preds = []
            for v in np.linspace(lo, hi, 20):
                probe = probe_base.copy()
                probe[feat] = v
                preds.append(float(pipe.predict(probe)[0]))
            assert max(preds) - min(preds) > 0, (
                f"{iso3} {year}: slider {feat} cannot move the prediction"
            )


def test_probe_feature_responsiveness_flags_inert_features():
    """A feature the model never uses must report zero spread."""
    meta, pipe, panel = _load_artifacts()
    feats = meta["feature_names"]
    train = app.get_training_data(panel, meta["train_end"])
    row = panel[(panel["iso3"] == "GHA") & (panel["year"] == 2019)].iloc[0]
    probe_base = pd.DataFrame([{f: row.get(f, np.nan) for f in feats}], columns=feats)

    resp = app.probe_feature_responsiveness(pipe, probe_base, train, feats)
    assert resp, "probe returned nothing"
    # Electricity access is never split on by the deployed model.
    if "EG.ELC.ACCS.ZS" in resp:
        assert resp["EG.ELC.ACCS.ZS"] == 0.0
    assert any(v > 0 for v in resp.values()), "no feature moves the prediction"


def test_get_scenario_features_falls_back_without_pipeline():
    """Callers without a pipeline still get a usable list (no crash)."""
    meta = {"feature_names": ["EG.ELC.ACCS.ZS", "IT.NET.USER.ZS",
                              "NE.GDI.TOTL.ZS", "NE.TRD.GNFS.ZS",
                              "FP.CPI.TOTL.ZG", "SP.DYN.LE00.IN",
                              "NY.GDP.PCAP.CD"]}
    out = app.get_scenario_features(meta, [])
    assert 3 <= len(out) <= 5
    assert set(out) <= set(meta["feature_names"])


# ----------------------------------------------------------------------------
# B13: the contributions table must sort without raising
# ----------------------------------------------------------------------------

def test_contribution_table_sorts_by_magnitude_not_string():
    """B13 regression: pre-formatted strings made sort_values(key=abs) raise
    TypeError: bad operand type for abs(): 'str'."""
    numeric = pd.DataFrame([
        {"Feature": "a", "Individual effect (pp)": -0.5},
        {"Feature": "b", "Individual effect (pp)": 0.2},
        {"Feature": "c", "Individual effect (pp)": 0.9},
    ])
    out = numeric.sort_values("Individual effect (pp)",
                              key=lambda s: s.abs(), ascending=False)
    assert list(out["Feature"]) == ["c", "a", "b"]

    stringly = numeric.copy()
    stringly["Individual effect (pp)"] = stringly["Individual effect (pp)"].map(
        lambda v: f"{v:+.3f}")
    with pytest.raises(TypeError):
        stringly.sort_values("Individual effect (pp)", key=abs, ascending=False)


def test_contribution_effects_are_numeric_in_source():
    """The app must not re-introduce string formatting in that column."""
    src = (REPO_ROOT / "app.py").read_text(encoding="utf-8")
    assert '"Model response (pp)": f"' not in src, (
        "Model response column is being formatted as a string again (B13)")
    assert 'key=abs' not in src, "sort_values(key=abs) re-introduced (B13)"


# ----------------------------------------------------------------------------
# Terracotta Editorial design system: non-negotiables
# ----------------------------------------------------------------------------

DESIGN_FILES = ["app.py", "src/ui.py", "src/theme.py", "src/visualization.py"]

EMOJI_RANGES = (
    (0x1F300, 0x1FAFF),
    (0x2600, 0x27BF),
    (0x1F000, 0x1F2FF),
    (0x2190, 0x21FF),
    (0xFE0F, 0xFE0F),
)


@pytest.mark.parametrize("rel", DESIGN_FILES)
def test_no_emoji_anywhere_in_the_interface(rel):
    """Spec section 43: no emoji in titles, navigation, labels or copy."""
    text = (REPO_ROOT / rel).read_text(encoding="utf-8")
    offenders = {
        ch for ch in text
        if any(lo <= ord(ch) <= hi for lo, hi in EMOJI_RANGES)
    }
    assert not offenders, f"{rel} contains emoji: {offenders}"


@pytest.mark.parametrize("rel", DESIGN_FILES)
def test_no_em_dashes(rel):
    """Spec section 46: em dashes are not used anywhere in the product."""
    text = (REPO_ROOT / rel).read_text(encoding="utf-8")
    assert "\u2014" not in text, f"{rel} contains an em dash"


def test_no_generic_marketing_language():
    """Spec sections 44 and 45: no SaaS marketing verbs in the copy."""
    text = (REPO_ROOT / "app.py").read_text(encoding="utf-8").lower()
    for word in ["unlock", "empower", "supercharge", "revolutioniz",
                 "seamlessly", "cutting-edge", "next-generation",
                 "ai-powered", "game-changing", "at a glance",
                 "dive in", "powered by"]:
        assert word not in text, f"marketing language in app copy: {word}"


def test_app_uses_the_editorial_components_not_default_streamlit_chrome():
    """Headings and metrics come from the design system, not st defaults."""
    src = (REPO_ROOT / "app.py").read_text(encoding="utf-8")
    for banned in ["st.title(", "st.header(", "st.subheader(", "st.metric(",
                   "st.warning(", "st.info("]:
        assert banned not in src, f"{banned} bypasses the design system"
    assert "inject_editorial_styles()" in src
    assert "ui.page_header(" in src and "ui.kpi_grid(" in src


def test_navigation_labels_are_unchanged_and_plain():
    """Spec section 42: keep the four analytical page names, no additions."""
    assert app.PAGES == [
        "Project Overview",
        "Explore Africa",
        "Model Performance",
        "Scenario Explorer",
    ]


def test_causal_guardrail_preserves_the_original_substance():
    """Spec sections 29 and 58: the caveat may be restyled, never weakened."""
    src = (REPO_ROOT / "app.py").read_text(encoding="utf-8")
    for phrase in [
        "not for causal policy-effect estimation",
        "statistical associations",
        "cannot prove",
        "not causal estimates",
    ]:
        assert phrase in src, f"causal guardrail lost the phrase: {phrase}"


def test_split_summary_reads_from_metadata():
    meta = {"split_target_years": {"train": [2001, 2018], "val": [2019, 2021],
                                   "test": [2022, 2024]}}
    text = app.split_summary(meta)
    assert "2001" in text and "2024" in text
    assert "\u2014" not in text


def test_format_value_handles_missing_numbers():
    assert app.format_value(3.14159) == "3.1"
    assert app.format_value(float("nan")) == "Not observed"
    assert app.format_value(None, missing="-") == "-"


# ----------------------------------------------------------------------------
# Page smoke tests: every page must render without raising
# ----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def app_test():
    """Run the application headlessly with Streamlit's AppTest harness."""
    from streamlit.testing.v1 import AppTest

    harness = AppTest.from_file(str(REPO_ROOT / "app.py"), default_timeout=300)
    return harness.run()


def test_overview_page_renders(app_test):
    assert not app_test.exception
    assert [b.label for b in app_test.button] == app.PAGES
    rendered = " ".join(m.value for m in app_test.markdown)
    assert "Africa" in rendered and "Growth Explorer" in rendered
    assert "ed-kpi-value" in rendered          # editorial KPI system in use
    assert "ed-process-step" in rendered       # how it works strip


@pytest.mark.parametrize("page", ["Explore Africa", "Model Performance",
                                  "Scenario Explorer"])
def test_other_pages_render(app_test, page):
    app_test.button(key=f"nav_{page}").click().run()
    assert not app_test.exception, [str(e.value) for e in app_test.exception]
    rendered = " ".join(m.value for m in app_test.markdown)
    assert "ed-section-title" in rendered


def test_scenario_page_keeps_its_analytical_guardrails(app_test):
    app_test.button(key="nav_Scenario Explorer").click().run()
    assert not app_test.exception
    assert 3 <= len(app_test.slider) <= 5, "scenario controls must stay 3 to 5"
    rendered = " ".join(m.value for m in app_test.markdown)
    assert "ed-guardrail" in rendered, "causal guardrail missing"
    assert "training" in rendered.lower(), "training-window guardrail copy lost"
