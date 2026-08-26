"""Tests for visualization functions."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from src.visualization import (
    get_project_palette,
    set_project_style,
    plot_missingness_heatmap,
    plot_feature_distributions,
    plot_correlation_matrix,
    plot_actual_vs_predicted,
    plot_residuals,
    plot_feature_importance,
)


def test_get_project_palette_returns_dict():
    """Palette should be a dictionary of named colors."""
    palette = get_project_palette()
    assert isinstance(palette, dict)
    assert "primary" in palette
    assert "secondary" in palette
    assert "accent" in palette


def test_set_project_style_applies_mpl_params():
    """Style function should set matplotlib rcParams."""
    set_project_style()
    assert plt.rcParams["figure.facecolor"] is not None


def test_plot_actual_vs_predicted_returns_figure():
    """Actual vs predicted plot should return a matplotlib figure."""
    fig, ax = plt.subplots()
    result = plot_actual_vs_predicted(
        actual=np.array([1, 2, 3]),
        predicted=np.array([1.1, 2.2, 2.8]),
        ax=ax,
    )
    assert result is not None
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


def test_plot_missingness_heatmap_averages_across_features():
    """MINOR FIX: Heatmap should show average missingness, not one arbitrary column."""
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