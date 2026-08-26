"""Visualization functions with consistent project visual language.

All charts use the same color palette, font settings, and styling for a
professional, unified look across the application and report.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)

PROJECT_PALETTE = {
    "primary": "#1B4F72",
    "secondary": "#2E86C1",
    "accent": "#E74C3C",
    "positive": "#27AE60",
    "negative": "#C0392B",
    "neutral": "#7F8C8D",
    "background": "#F8F9FA",
    "grid": "#DEE2E6",
    "text": "#2C3E50",
}


def get_project_palette() -> Dict[str, str]:
    """Return the project color palette dictionary.

    Returns:
        Dictionary mapping color names to hex values.
    """
    return PROJECT_PALETTE.copy()


def set_project_style() -> None:
    """Apply consistent matplotlib styling for all project charts."""
    plt.rcParams.update({
        "figure.facecolor": PROJECT_PALETTE["background"],
        "axes.facecolor": "white",
        "axes.edgecolor": PROJECT_PALETTE["grid"],
        "axes.grid": True,
        "grid.color": PROJECT_PALETTE["grid"],
        "grid.alpha": 0.7,
        "text.color": PROJECT_PALETTE["text"],
        "xtick.color": PROJECT_PALETTE["text"],
        "ytick.color": PROJECT_PALETTE["text"],
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "figure.titlesize": 16,
    })
    logger.info("Project matplotlib style applied")


def plot_missingness_heatmap(
    df: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    title: str = "Missing Data by Country and Year",
) -> plt.Figure:
    """Plot a heatmap of average missingness across all feature columns.

    MINOR FIX: Averages missingness across all numeric feature columns
    instead of visualizing one arbitrary column.

    Args:
        df: DataFrame with columns including 'iso3', 'year', and feature columns.
        ax: Matplotlib axes to plot on. If None, creates new figure.
        title: Plot title.

    Returns:
        Matplotlib Figure object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        fig = ax.get_figure()

    # Average missingness across all feature columns per country-year
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                   if c not in ["year"]]
    if not feature_cols:
        logger.warning("No numeric feature columns found for missingness heatmap")
        return fig

    missing_by_group = df.groupby(["iso3", "year"])[feature_cols].apply(
        lambda x: x.isnull().mean().mean()  # Average across features
    ).reset_index(name="avg_missingness")

    pivot = missing_by_group.pivot_table(
        index="iso3", columns="year", values="avg_missingness", aggfunc="first"
    )

    sns.heatmap(
        pivot, ax=ax, cmap="YlOrRd", vmin=0, vmax=1,
        cbar_kws={"label": "Average missing fraction across features"},
        linewidths=0.5,
    )
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel("Country")
    fig.tight_layout()
    return fig


def plot_feature_distributions(
    df: pd.DataFrame,
    feature_cols: list,
    ncols: int = 3,
    title: str = "Feature Distributions",
) -> plt.Figure:
    """Plot histograms of feature distributions.

    Args:
        df: DataFrame containing the features.
        feature_cols: List of feature column names to plot.
        ncols: Number of columns in the subplot grid.
        title: Overall figure title.

    Returns:
        Matplotlib Figure object.
    """
    nrows = (len(feature_cols) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if nrows * ncols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, col in enumerate(feature_cols):
        if col in df.columns:
            axes[i].hist(df[col].dropna(), bins=30, color=PROJECT_PALETTE["secondary"],
                        alpha=0.7, edgecolor="white")
            axes[i].set_title(col, fontsize=10)
            axes[i].set_ylabel("Count")

    for j in range(len(feature_cols), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


def plot_correlation_matrix(
    df: pd.DataFrame,
    feature_cols: list,
    ax: Optional[plt.Axes] = None,
    title: str = "Feature Correlation Matrix",
) -> plt.Figure:
    """Plot a correlation heatmap of features.

    Args:
        df: DataFrame containing the features.
        feature_cols: List of feature column names.
        ax: Matplotlib axes. If None, creates new figure.
        title: Plot title.

    Returns:
        Matplotlib Figure object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.get_figure()

    corr = df[feature_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, ax=ax, cmap="RdBu_r", center=0,
        vmin=-1, vmax=1, annot=True, fmt=".2f",
        square=True, linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_actual_vs_predicted(
    actual: np.ndarray,
    predicted: np.ndarray,
    ax: Optional[plt.Axes] = None,
    title: str = "Actual vs. Predicted GDP Growth",
) -> plt.Figure:
    """Plot actual vs predicted values with identity line.

    Args:
        actual: Array of actual target values.
        predicted: Array of predicted values.
        ax: Matplotlib axes. If None, creates new figure.
        title: Plot title.

    Returns:
        Matplotlib Figure object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.get_figure()

    ax.scatter(actual, predicted, alpha=0.5, color=PROJECT_PALETTE["secondary"],
              edgecolors="white", linewidth=0.5, s=40)

    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())
    ax.plot([min_val, max_val], [min_val, max_val],
           color=PROJECT_PALETTE["accent"], linestyle="--", linewidth=1.5,
           label="Perfect prediction")

    ax.set_xlabel("Actual GDP Growth (%)")
    ax.set_ylabel("Predicted GDP Growth (%)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_residuals(
    actual: np.ndarray,
    predicted: np.ndarray,
    ax: Optional[plt.Axes] = None,
    title: str = "Residual Analysis",
) -> plt.Figure:
    """Plot residuals vs predicted values.

    Args:
        actual: Array of actual target values.
        predicted: Array of predicted values.
        ax: Matplotlib axes. If None, creates new figure.
        title: Plot title.

    Returns:
        Matplotlib Figure object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()

    residuals = actual - predicted
    ax.scatter(predicted, residuals, alpha=0.5, color=PROJECT_PALETTE["secondary"],
              edgecolors="white", linewidth=0.5, s=40)
    ax.axhline(y=0, color=PROJECT_PALETTE["accent"], linestyle="--", linewidth=1.5)
    ax.set_xlabel("Predicted GDP Growth (%)")
    ax.set_ylabel("Residual (Actual - Predicted)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_feature_importance(
    importance: pd.Series,
    ax: Optional[plt.Axes] = None,
    title: str = "Feature Importance",
    top_n: int = 15,
) -> plt.Figure:
    """Plot horizontal bar chart of feature importance.

    Args:
        importance: Series with feature names as index and importance as values.
        ax: Matplotlib axes. If None, creates new figure.
        title: Plot title.
        top_n: Number of top features to display.

    Returns:
        Matplotlib Figure object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    top_features = importance.sort_values(ascending=True).tail(top_n)
    ax.barh(top_features.index, top_features.values,
           color=PROJECT_PALETTE["secondary"], edgecolor="white")
    ax.set_xlabel("Importance")
    ax.set_title(title)
    fig.tight_layout()
    return fig