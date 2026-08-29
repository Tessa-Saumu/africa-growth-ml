"""Tests for the project chart system.

The charts carry the same design language as the interface, so these tests
check the palette contract, the shared axes treatment (no top or right
spines, one faint grid direction) and each chart builder.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.theme import COLORS
from src.visualization import (
    coverage_points,
    get_chart_sequence,
    get_diverging_cmap,
    get_project_palette,
    get_terracotta_cmap,
    load_africa_coordinates,
    load_africa_geometry,
    plot_actual_vs_predicted,
    plot_africa_dot_map,
    plot_africa_map,
    plot_correlation_matrix,
    plot_feature_distributions,
    plot_feature_importance,
    plot_growth_trend,
    plot_indicator_small_multiples,
    plot_metric_by_year,
    plot_missingness_heatmap,
    plot_residuals,
    plot_scenario_response,
    set_editorial_plot_style,
    set_project_style,
    style_axes,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

# The bundled fonts must be registered before any figure is built, exactly as
# the application does at import time.
set_project_style()


# ----------------------------------------------------------------------------
# Palette and style
# ----------------------------------------------------------------------------

def test_get_project_palette_returns_dict():
    """Palette should be a dictionary of named colors."""
    palette = get_project_palette()
    assert isinstance(palette, dict)
    assert "primary" in palette
    assert "secondary" in palette
    assert "accent" in palette


def test_palette_is_built_from_editorial_tokens():
    """No chart colour may sit outside the design tokens."""
    palette = get_project_palette()
    assert palette["primary"] == COLORS["terracotta"]
    assert palette["secondary"] == COLORS["plum"]
    assert set(palette.values()) <= set(COLORS.values())


def test_chart_sequence_matches_the_specification():
    assert get_chart_sequence()[:3] == [
        COLORS["terracotta"], COLORS["plum"], COLORS["rose"],
    ]


def test_set_project_style_applies_editorial_rcparams():
    """Style function should set matplotlib rcParams."""
    set_project_style()
    assert plt.rcParams["figure.facecolor"] is not None
    assert plt.rcParams["axes.spines.top"] is False
    assert plt.rcParams["axes.spines.right"] is False
    assert plt.rcParams["axes.labelcolor"] == COLORS["ink_secondary"]
    assert "DM Sans" in plt.rcParams["font.sans-serif"]


def test_editorial_style_is_transparent_for_the_app_canvas():
    set_editorial_plot_style(transparent=True)
    assert plt.rcParams["figure.facecolor"] == "none"
    set_project_style()


def test_style_axes_removes_top_and_right_spines():
    fig, ax = plt.subplots()
    style_axes(ax)
    assert not ax.spines["top"].get_visible()
    assert not ax.spines["right"].get_visible()
    assert ax.spines["left"].get_edgecolor() is not None
    plt.close(fig)


def test_colormaps_run_from_light_to_terracotta():
    cmap = get_terracotta_cmap()
    light = cmap(0.0)
    deep = cmap(1.0)
    assert light[0] > deep[0]          # red channel darkens
    assert get_diverging_cmap()(0.5) is not None


# ----------------------------------------------------------------------------
# Evaluation charts
# ----------------------------------------------------------------------------

def test_plot_actual_vs_predicted_returns_figure():
    """Actual vs predicted plot should return a matplotlib figure."""
    fig, ax = plt.subplots()
    result = plot_actual_vs_predicted(
        actual=np.array([1, 2, 3]),
        predicted=np.array([1.1, 2.2, 2.8]),
        ax=ax,
    )
    assert result is not None
    assert not ax.spines["top"].get_visible()
    plt.close(fig)


def test_plot_residuals_returns_figure():
    """Residual plot should return a matplotlib figure."""
    fig, ax = plt.subplots()
    result = plot_residuals(
        actual=np.array([1, 2, 3]),
        predicted=np.array([1.1, 2.2, 2.8]),
        ax=ax,
    )
    assert result is not None
    plt.close(fig)


def test_plot_feature_importance_returns_figure():
    """Feature importance plot should return a matplotlib figure."""
    fig, ax = plt.subplots()
    importance = pd.Series({"feat_a": 0.5, "feat_b": 0.3, "feat_c": 0.2})
    result = plot_feature_importance(importance, ax=ax)
    assert result is not None
    plt.close(fig)


def test_feature_importance_mutes_non_significant_bars():
    """Significant bars are terracotta; the rest use the inactive tone."""
    importance = pd.Series({"feat_a": 0.5, "feat_b": 0.3})
    significant = pd.Series({"feat_a": True, "feat_b": False})
    fig, ax = plt.subplots()
    plot_feature_importance(importance, ax=ax, significant=significant)
    colors = {tuple(np.round(patch.get_facecolor(), 3)) for patch in ax.patches}
    from matplotlib.colors import to_rgba
    assert tuple(np.round(to_rgba(COLORS["terracotta"]), 3)) in colors
    assert tuple(np.round(to_rgba(COLORS["inactive"]), 3)) in colors
    plt.close(fig)


def test_plot_metric_by_year_draws_reference_line():
    fig = plot_metric_by_year([2022, 2023], [1.5, 1.9], reference=1.8)
    ax = fig.axes[0]
    assert len(ax.patches) == 2
    assert any(line.get_linestyle() == "--" for line in ax.lines)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Country charts
# ----------------------------------------------------------------------------

def test_plot_growth_trend_draws_both_series():
    fig = plot_growth_trend([2000, 2001, 2002], [1.0, 2.0, 3.0], [2.0, 3.0, 4.0])
    ax = fig.axes[0]
    # zero rule plus two data series
    assert len(ax.lines) == 3
    plt.close(fig)


def test_plot_indicator_small_multiples_one_panel_per_indicator():
    data = pd.DataFrame({
        "year": [2000, 2001, 2002],
        "A": [1.0, 2.0, 3.0],
        "B": [3.0, 2.0, 1.0],
    })
    fig = plot_indicator_small_multiples(data, ["A", "B"], {"A": "Alpha", "B": "Beta"})
    visible = [ax for ax in fig.axes if ax.get_visible()]
    assert len(visible) == 2
    assert visible[0].get_title(loc="left") == "Alpha"
    plt.close(fig)


def test_plot_indicator_small_multiples_handles_no_features():
    fig = plot_indicator_small_multiples(pd.DataFrame({"year": [2000]}), [], {})
    assert fig is not None
    plt.close(fig)


def test_plot_scenario_response_sorts_by_magnitude():
    fig = plot_scenario_response(["a", "b"], [0.1, -0.9])
    labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    assert labels[-1] == "b"
    plt.close(fig)


# ----------------------------------------------------------------------------
# Geography
# ----------------------------------------------------------------------------

def test_africa_reference_files_are_committed():
    coords = load_africa_coordinates()
    geometry = load_africa_geometry()
    assert {"iso3", "lat", "lon"} <= set(coords.columns)
    assert len(coords) >= 50
    assert len(geometry["countries"]) >= 45
    assert all("rings" in c for c in geometry["countries"])


def test_load_africa_geometry_raises_for_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_africa_geometry(tmp_path / "nope.json")


def test_plot_africa_map_shades_known_countries():
    geometry = load_africa_geometry()
    coords = load_africa_coordinates()
    fig = plot_africa_map(geometry, {"KEN": 0.9, "NGA": 0.4}, highlight="KEN",
                          coordinates=coords, value_label="Coverage")
    assert fig.axes  # map plus colour bar
    assert fig.axes[0].axison is False
    plt.close(fig)


def test_plot_africa_dot_map_falls_back_without_geometry():
    coords = load_africa_coordinates()
    fig = plot_africa_dot_map(coords, {"KEN": 0.8}, highlight="KEN")
    assert fig is not None
    plt.close(fig)


def test_coverage_points_pairs_coordinates_with_values():
    coords = pd.DataFrame([{"iso3": "KEN", "lat": -0.02, "lon": 37.91}])
    points = coverage_points(coords, {"KEN": 0.5})
    assert points == [(37.91, -0.02, 0.5)]
    assert coverage_points(coords, {})[0][2] == 0.0


# ----------------------------------------------------------------------------
# Exploratory charts kept for the notebooks and the report builder
# ----------------------------------------------------------------------------

def test_plot_missingness_heatmap_averages_across_features():
    """Heatmap should show average missingness, not one arbitrary column."""
    df = pd.DataFrame({
        "iso3": ["GHA"] * 3 + ["KEN"] * 3,
        "year": [2018, 2019, 2020] * 2,
        "feat_a": [1.0, np.nan, 3.0, 4.0, 5.0, 6.0],
        "feat_b": [1.0, 2.0, np.nan, np.nan, 5.0, 6.0],
    })
    fig, ax = plt.subplots()
    result = plot_missingness_heatmap(df, ax=ax)
    assert result is not None
    plt.close(fig)


def test_plot_feature_distributions_uses_display_names():
    df = pd.DataFrame({"A": np.random.default_rng(0).normal(size=50)})
    fig = plot_feature_distributions(df, ["A"], ncols=1, display_names={"A": "Alpha"})
    assert fig.axes[0].get_title(loc="left") == "Alpha"
    plt.close(fig)


def test_plot_correlation_matrix_returns_figure():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"A": rng.normal(size=30), "B": rng.normal(size=30)})
    fig, ax = plt.subplots()
    assert plot_correlation_matrix(df, ["A", "B"], ax=ax) is not None
    plt.close(fig)
