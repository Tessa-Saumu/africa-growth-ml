"""Chart system for the Africa Growth Explorer.

One plotting language for the whole project: the Terracotta Editorial tokens
from :mod:`src.theme`, DM Sans typography, no top or right spines, a nearly
invisible horizontal grid, and one or two colours per chart. The same
functions serve the Streamlit application, the notebooks and the report
asset builder, so every figure in the project looks like it came from the
same publication.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from src.theme import (
    CHART_COLORS,
    COLORS,
    MAP_SCALE,
    register_project_fonts,
)

logger = logging.getLogger(__name__)

# Chart-facing names kept stable for existing callers (notebooks, report
# builder), remapped onto the editorial tokens.
PROJECT_PALETTE: Dict[str, str] = {
    "primary": COLORS["terracotta"],
    "secondary": COLORS["plum"],
    "accent": COLORS["terracotta_deep"],
    "tertiary": COLORS["rose"],
    "context": COLORS["sand"],
    "positive": COLORS["positive"],
    "negative": COLORS["negative"],
    "neutral": COLORS["olive"],
    "inactive": COLORS["inactive"],
    "background": COLORS["canvas"],
    "surface": COLORS["surface"],
    "grid": COLORS["border"],
    "text": COLORS["ink"],
    "text_secondary": COLORS["ink_secondary"],
    "muted": COLORS["muted"],
}

FONT_STACK: List[str] = ["DM Sans", "Arial", "Helvetica", "DejaVu Sans"]

_FONTS_REGISTERED = False


def get_project_palette() -> Dict[str, str]:
    """Return the project color palette dictionary.

    Returns:
        Dictionary mapping color names to hex values.
    """
    return PROJECT_PALETTE.copy()


def get_chart_sequence() -> List[str]:
    """Return the fixed categorical chart color sequence.

    Returns:
        List of hex colors, primary series first.
    """
    return list(CHART_COLORS)


def get_terracotta_cmap() -> LinearSegmentedColormap:
    """Sequential terracotta colormap for maps and density displays.

    Returns:
        Matplotlib colormap running light tint to deep terracotta.
    """
    return LinearSegmentedColormap.from_list(
        "terracotta", [COLORS["canvas"], *MAP_SCALE], N=256
    )


def get_diverging_cmap() -> LinearSegmentedColormap:
    """Diverging colormap for correlation-style displays.

    Returns:
        Matplotlib colormap running plum through canvas to terracotta.
    """
    return LinearSegmentedColormap.from_list(
        "plum_terracotta",
        [COLORS["plum"], COLORS["rose"], COLORS["canvas"],
         COLORS["terracotta_light"], COLORS["terracotta_deep"]],
        N=256,
    )


def set_project_style(transparent: bool = False) -> None:
    """Apply the editorial Matplotlib style to every subsequent figure.

    Args:
        transparent: When True, figures are drawn without a background so
            charts sit directly on the application canvas. When False, the
            warm canvas colour is used, which suits exported PNGs.

    Returns:
        None.
    """
    global _FONTS_REGISTERED
    if not _FONTS_REGISTERED:
        register_project_fonts()
        _FONTS_REGISTERED = True

    face = "none" if transparent else COLORS["canvas"]
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": FONT_STACK,
        "font.size": 10,
        "axes.titlesize": 14,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.frameon": False,
        "figure.facecolor": face,
        "savefig.facecolor": face,
        "axes.facecolor": "none",
        "axes.edgecolor": COLORS["border_strong"],
        "axes.linewidth": 0.8,
        "axes.labelcolor": COLORS["ink_secondary"],
        "axes.titlecolor": COLORS["ink"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.prop_cycle": plt.cycler(color=CHART_COLORS),
        "grid.color": COLORS["border_strong"],
        "grid.alpha": 0.35,
        "grid.linewidth": 0.7,
        "text.color": COLORS["ink"],
        "xtick.color": COLORS["ink_secondary"],
        "ytick.color": COLORS["ink_secondary"],
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "figure.dpi": 130,
        "figure.titlesize": 15,
        "lines.linewidth": 2.2,
        "lines.solid_capstyle": "round",
        "patch.edgecolor": "none",
    })
    logger.info("Editorial matplotlib style applied (transparent=%s)", transparent)


def set_editorial_plot_style(transparent: bool = True) -> None:
    """Apply the editorial style tuned for the Streamlit canvas.

    Args:
        transparent: Draw figures without a background rectangle.

    Returns:
        None.
    """
    set_project_style(transparent=transparent)


def style_axes(ax: plt.Axes, grid_axis: Optional[str] = "y") -> plt.Axes:
    """Finish an axes so every chart in the project matches.

    Removes the top and right spines, softens the remaining ones and draws a
    single faint grid direction behind the data.

    Args:
        ax: Axes to style.
        grid_axis: "y", "x", "both" or None to disable the grid.

    Returns:
        The same axes, for chaining.
    """
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(COLORS["border_strong"])
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(length=0, labelsize=10, colors=COLORS["ink_secondary"])
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=COLORS["border_strong"],
                alpha=0.35, linewidth=0.7)
        ax.set_axisbelow(True)
    else:
        ax.grid(False)
    return ax


def _title(ax: plt.Axes, title: str) -> None:
    """Apply the chart title convention: small, semibold, left aligned.

    Args:
        ax: Axes to title.
        title: Title text, or an empty string to omit it.

    Returns:
        None.
    """
    if title:
        ax.set_title(title, fontsize=14, fontweight="semibold", loc="left",
                     color=COLORS["ink"], pad=12)


def save_figure(fig: Figure, name: str, project_root: Optional[Path] = None,
                subdir: str = "figures", dpi: int = 160) -> Path:
    """Persist a figure to the project-wide figures/ folder.

    Single convention for every PNG the project produces (notebooks and the
    report-asset builder alike): ``<project_root>/figures/<name>.png``.

    Args:
        fig: Matplotlib figure to save.
        name: Descriptive file name ending in ``.png`` (e.g.
            "eda_missingness_heatmap.png").
        project_root: Repository root; resolved upward from cwd (looking for
            pyproject.toml) when omitted.
        subdir: Target folder name under the root.
        dpi: Rasterization density.

    Returns:
        Path of the written PNG.

    Raises:
        ValueError: If name does not end in .png.
    """
    if not name.endswith(".png"):
        raise ValueError(f"figure names must end in .png, got {name!r}")
    root = Path(project_root) if project_root else None
    if root is None:
        here = Path.cwd()
        root = next((q for q in [here, *here.parents]
                     if (q / "pyproject.toml").exists()), here)
    out_dir = root / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    logger.info("Saved figure: %s", path)
    return path


# ---------------------------------------------------------------------------
# Exploratory data displays
# ---------------------------------------------------------------------------

def plot_missingness_heatmap(
    df: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    title: str = "Missing data by country and year",
) -> plt.Figure:
    """Plot a heatmap of average missingness across all feature columns.

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

    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in ["year"]]
    if not feature_cols:
        logger.warning("No numeric feature columns found for missingness heatmap")
        return fig

    missing_by_group = df.groupby(["iso3", "year"])[feature_cols].apply(
        lambda x: x.isnull().mean().mean()
    ).reset_index(name="avg_missingness")

    pivot = missing_by_group.pivot_table(
        index="iso3", columns="year", values="avg_missingness", aggfunc="first"
    )

    sns.heatmap(
        pivot, ax=ax, cmap=get_terracotta_cmap(), vmin=0, vmax=1,
        cbar_kws={"label": "Average missing fraction across features"},
        linewidths=0.4, linecolor=COLORS["canvas"],
    )
    _title(ax, title)
    ax.set_xlabel("Year")
    ax.set_ylabel("Country")
    ax.grid(False)
    ax.tick_params(length=0, labelsize=9, colors=COLORS["ink_secondary"])
    fig.tight_layout()
    return fig


def plot_feature_distributions(
    df: pd.DataFrame,
    feature_cols: list,
    ncols: int = 3,
    title: str = "Feature distributions",
    display_names: Optional[Dict[str, str]] = None,
) -> plt.Figure:
    """Plot histograms of feature distributions as small multiples.

    Args:
        df: DataFrame containing the features.
        feature_cols: List of feature column names to plot.
        ncols: Number of columns in the subplot grid.
        title: Overall figure title.
        display_names: Optional code -> readable label mapping for subplot
            titles, so raw indicator codes stay out of the visible layer.

    Returns:
        Matplotlib Figure object.
    """
    display_names = display_names or {}
    nrows = (len(feature_cols) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.2 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, col in enumerate(feature_cols):
        ax = axes[i]
        if col in df.columns:
            ax.hist(df[col].dropna(), bins=30, color=COLORS["terracotta"],
                    alpha=0.85)
            ax.set_title(display_names.get(col, col), fontsize=11,
                         fontweight="semibold", loc="left", color=COLORS["ink"])
            ax.set_ylabel("Count")
        style_axes(ax)

    for j in range(len(feature_cols), len(axes)):
        axes[j].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=15, x=0.01, ha="left",
                     color=COLORS["ink"], fontweight="semibold")
    fig.tight_layout()
    return fig


def plot_correlation_matrix(
    df: pd.DataFrame,
    feature_cols: list,
    ax: Optional[plt.Axes] = None,
    title: str = "Feature correlation matrix",
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
        corr, mask=mask, ax=ax, cmap=get_diverging_cmap(), center=0,
        vmin=-1, vmax=1, annot=True, fmt=".2f", annot_kws={"size": 8},
        square=True, linewidths=0.4, linecolor=COLORS["canvas"],
        cbar_kws={"shrink": 0.8},
    )
    _title(ax, title)
    ax.tick_params(length=0, labelsize=9, colors=COLORS["ink_secondary"])
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Model evaluation displays
# ---------------------------------------------------------------------------

def plot_actual_vs_predicted(
    actual: np.ndarray,
    predicted: np.ndarray,
    ax: Optional[plt.Axes] = None,
    title: str = "Actual vs predicted next-year growth",
) -> plt.Figure:
    """Plot actual against predicted values with a reference diagonal.

    Args:
        actual: Array of actual target values (x axis).
        predicted: Array of predicted values (y axis).
        ax: Matplotlib axes. If None, creates new figure.
        title: Plot title.

    Returns:
        Matplotlib Figure object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.4, 5.2))
    else:
        fig = ax.get_figure()

    ax.scatter(actual, predicted, alpha=0.65, color=COLORS["terracotta"],
               s=34, linewidths=0)

    min_val = float(min(np.min(actual), np.min(predicted)))
    max_val = float(max(np.max(actual), np.max(predicted)))
    ax.plot([min_val, max_val], [min_val, max_val],
            color=COLORS["muted"], linestyle="--", linewidth=1)

    ax.set_xlabel("Actual growth (%)")
    ax.set_ylabel("Predicted growth (%)")
    _title(ax, title)
    style_axes(ax, grid_axis="both")
    fig.tight_layout()
    return fig


def plot_residuals(
    actual: np.ndarray,
    predicted: np.ndarray,
    ax: Optional[plt.Axes] = None,
    title: str = "Residuals against predicted growth",
) -> plt.Figure:
    """Plot residuals against predicted values.

    Args:
        actual: Array of actual target values.
        predicted: Array of predicted values.
        ax: Matplotlib axes. If None, creates new figure.
        title: Plot title.

    Returns:
        Matplotlib Figure object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
    else:
        fig = ax.get_figure()

    residuals = np.asarray(actual) - np.asarray(predicted)
    ax.scatter(predicted, residuals, alpha=0.6, color=COLORS["plum"], s=34,
               linewidths=0)
    ax.axhline(y=0, color=COLORS["terracotta"], linestyle="-", linewidth=1.4)
    ax.set_xlabel("Predicted growth (%)")
    ax.set_ylabel("Residual (actual minus predicted)")
    _title(ax, title)
    style_axes(ax, grid_axis="y")
    fig.tight_layout()
    return fig


def plot_feature_importance(
    importance: pd.Series,
    ax: Optional[plt.Axes] = None,
    title: str = "Permutation importance",
    top_n: int = 15,
    significant: Optional[pd.Series] = None,
) -> plt.Figure:
    """Plot horizontal bars of feature importance, sorted descending.

    Args:
        importance: Series indexed by feature label with importance values.
        ax: Matplotlib axes. If None, creates new figure.
        title: Plot title.
        top_n: Number of features to display.
        significant: Optional boolean Series aligned to ``importance``.
            Features flagged False are drawn in the inactive tone so the
            significant ones visually dominate; no other encoding changes.

    Returns:
        Matplotlib Figure object.
    """
    top_features = importance.sort_values(ascending=True).tail(top_n)

    if ax is None:
        height = max(2.4, 0.42 * len(top_features) + 1.1)
        fig, ax = plt.subplots(figsize=(8.4, height))
    else:
        fig = ax.get_figure()

    if significant is not None:
        flags = significant.reindex(top_features.index).fillna(False)
        colors = [COLORS["terracotta"] if bool(flag) else COLORS["inactive"]
                  for flag in flags]
    else:
        colors = [COLORS["terracotta"]] * len(top_features)

    ax.barh(list(top_features.index), list(top_features.values), color=colors,
            height=0.62)
    ax.axvline(0, color=COLORS["border_strong"], linewidth=0.8)
    ax.set_xlabel("Increase in validation error when the column is shuffled")
    _title(ax, title)
    style_axes(ax, grid_axis="x")
    ax.tick_params(axis="y", labelsize=10)
    fig.tight_layout()
    return fig


def plot_metric_by_year(
    years: Sequence[int],
    values: Sequence[float],
    reference: Optional[float] = None,
    ax: Optional[plt.Axes] = None,
    title: str = "Error by reference year",
    ylabel: str = "Mean absolute error (pp)",
    reference_label: str = "Baseline",
) -> plt.Figure:
    """Plot a compact per-year error bar chart with an optional reference.

    Args:
        years: Reference years on the x axis.
        values: Metric value per year.
        reference: Optional horizontal reference value, for example the
            global-mean baseline error.
        ax: Matplotlib axes. If None, creates new figure.
        title: Plot title.
        ylabel: Y axis label.
        reference_label: Label drawn next to the reference line.

    Returns:
        Matplotlib Figure object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.4, 3.2))
    else:
        fig = ax.get_figure()

    ax.bar([str(y) for y in years], list(values), color=COLORS["terracotta"],
           width=0.45)
    if reference is not None:
        ax.axhline(reference, color=COLORS["plum"], linestyle="--",
                   linewidth=1.2)
        ax.annotate(
            f"{reference_label} {reference:.2f}",
            xy=(0.995, reference), xycoords=("axes fraction", "data"),
            ha="right", va="bottom", fontsize=10, color=COLORS["plum"],
        )
    ax.set_ylabel(ylabel)
    _title(ax, title)
    style_axes(ax, grid_axis="y")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Country displays
# ---------------------------------------------------------------------------

def plot_growth_trend(
    years: Sequence[int],
    observed: Sequence[float],
    next_year: Optional[Sequence[float]] = None,
    ax: Optional[plt.Axes] = None,
    title: str = "GDP per capita growth",
) -> plt.Figure:
    """Plot observed growth over time with the next-year target series.

    Args:
        years: Reference years.
        observed: GDP per capita growth observed in each year (%).
        next_year: Optional next-year growth, the modelling target.
        ax: Matplotlib axes. If None, creates new figure.
        title: Plot title.

    Returns:
        Matplotlib Figure object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(9.6, 3.8))
    else:
        fig = ax.get_figure()

    ax.axhline(0, color=COLORS["border_strong"], linewidth=0.9)
    ax.plot(list(years), list(observed), color=COLORS["terracotta"],
            linewidth=2.2, label="Observed growth in year t")
    if next_year is not None:
        ax.plot(list(years), list(next_year), color=COLORS["plum"],
                linewidth=1.6, linestyle="--",
                label="Growth in year t+1 (model target)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Annual growth (%)")
    _title(ax, title)
    style_axes(ax, grid_axis="y")
    leg = ax.legend(loc="upper left", frameon=False, fontsize=10,
                    bbox_to_anchor=(0, 1.02), ncols=2)
    for text in leg.get_texts():
        text.set_color(COLORS["ink_secondary"])
    fig.tight_layout()
    return fig


def plot_indicator_small_multiples(
    data: pd.DataFrame,
    features: Sequence[str],
    display_names: Dict[str, str],
    year_col: str = "year",
    ncols: int = 3,
) -> plt.Figure:
    """Plot one small chart per indicator instead of one crowded chart.

    Indicators live on different scales, so a shared axis misleads. Small
    multiples keep one colour and one story per panel.

    Args:
        data: Country rows containing ``year_col`` and the feature columns.
        features: Feature codes to plot, in display order.
        display_names: Feature code -> readable label.
        year_col: Name of the year column.
        ncols: Panels per row.

    Returns:
        Matplotlib Figure object.
    """
    features = [f for f in features if f in data.columns]
    if not features:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.set_axis_off()
        return fig

    nrows = (len(features) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 2.6 * nrows),
                             squeeze=False)
    flat = axes.reshape(-1)

    for i, feat in enumerate(features):
        ax = flat[i]
        series = data[[year_col, feat]].dropna()
        ax.plot(series[year_col], series[feat], color=COLORS["terracotta"],
                linewidth=2.0)
        ax.set_title(display_names.get(feat, feat), fontsize=11,
                     fontweight="semibold", loc="left", color=COLORS["ink"])
        style_axes(ax, grid_axis="y")
        ax.tick_params(labelsize=9)

    for j in range(len(features), len(flat)):
        flat[j].set_visible(False)

    fig.tight_layout(h_pad=2.2, w_pad=2.0)
    return fig


def plot_africa_dot_map(
    coordinates: pd.DataFrame,
    values: Dict[str, float],
    highlight: Optional[str] = None,
    title: str = "",
    value_label: str = "",
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plot African countries as a terracotta-shaded dot field.

    One dot per country, placed at its centroid on an equirectangular
    projection and shaded on the terracotta tonal scale. Deliberately not a
    rainbow choropleth and not a stock illustration.

    Args:
        coordinates: Frame with columns iso3, lat, lon.
        values: iso3 -> value in [0, 1]; countries without a value are drawn
            in the inactive tone.
        highlight: Optional iso3 drawn with a deep terracotta ring.
        title: Plot title.
        value_label: Colour bar label, for example "Indicator coverage".
        ax: Matplotlib axes. If None, creates new figure.

    Returns:
        Matplotlib Figure object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.2, 6.4))
    else:
        fig = ax.get_figure()

    cmap = get_terracotta_cmap()
    known = coordinates[coordinates["iso3"].isin(values.keys())]
    unknown = coordinates[~coordinates["iso3"].isin(values.keys())]

    if not unknown.empty:
        ax.scatter(unknown["lon"], unknown["lat"], s=110,
                   color=COLORS["inactive"], linewidths=0, zorder=2)

    scatter = None
    if not known.empty:
        colour_values = [values[iso] for iso in known["iso3"]]
        scatter = ax.scatter(
            known["lon"], known["lat"], c=colour_values, cmap=cmap,
            vmin=0.0, vmax=1.0, s=150, linewidths=0.6,
            edgecolors=COLORS["canvas"], zorder=3,
        )

    if highlight is not None and highlight in set(coordinates["iso3"]):
        row = coordinates[coordinates["iso3"] == highlight].iloc[0]
        ax.scatter([row["lon"]], [row["lat"]], s=330, facecolors="none",
                   edgecolors=COLORS["terracotta_deep"], linewidths=1.6,
                   zorder=4)

    ax.set_aspect("equal")
    ax.set_axis_off()
    _title(ax, title)

    if scatter is not None and value_label:
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.032, pad=0.02,
                            shrink=0.62)
        cbar.set_label(value_label, fontsize=11,
                       color=COLORS["ink_secondary"])
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(length=0, labelsize=10,
                            colors=COLORS["ink_secondary"])

    fig.tight_layout()
    return fig


def plot_africa_map(
    geometry: Dict[str, object],
    values: Dict[str, float],
    highlight: Optional[str] = None,
    coordinates: Optional[pd.DataFrame] = None,
    title: str = "",
    value_label: str = "",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Draw Africa as a terracotta tonal choropleth.

    Country outlines come from the committed simplified geometry, so no
    geospatial dependency is required. Countries that are too small to appear
    at this resolution (island states) are drawn as dots from the centroid
    reference, shaded on the same scale.

    Args:
        geometry: Payload from :func:`load_africa_geometry`.
        values: iso3 -> value. Countries without a value are drawn in the
            inactive tone.
        highlight: Optional iso3 outlined in ink so it reads against the fill.
        coordinates: Optional centroid frame used for island states.
        title: Plot title.
        value_label: Colour bar label.
        vmin: Lower bound of the colour scale. Defaults to the minimum of
            ``values``, so the tonal range spans the data actually present.
        vmax: Upper bound of the colour scale. Defaults to the maximum.
        ax: Matplotlib axes. If None, creates new figure.

    Returns:
        Matplotlib Figure object.
    """
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.0, 6.4))
    else:
        fig = ax.get_figure()

    cmap = get_terracotta_cmap()
    present = [v for v in values.values() if v is not None and np.isfinite(v)]
    lo = vmin if vmin is not None else (min(present) if present else 0.0)
    hi = vmax if vmax is not None else (max(present) if present else 1.0)
    if hi <= lo:
        hi = lo + 1e-6
    norm = Normalize(vmin=lo, vmax=hi)
    countries = list(geometry.get("countries", []))  # type: ignore[arg-type]
    drawn = set()

    for country in countries:
        iso3 = str(country["iso3"])
        drawn.add(iso3)
        value = values.get(iso3)
        face = cmap(norm(value)) if value is not None else COLORS["inactive"]
        edge = COLORS["ink"] if iso3 == highlight else COLORS["canvas"]
        width = 1.4 if iso3 == highlight else 0.6
        for ring in country["rings"]:  # type: ignore[index]
            xs = [point[0] for point in ring]
            ys = [point[1] for point in ring]
            ax.fill(xs, ys, facecolor=face, edgecolor=edge, linewidth=width,
                    zorder=3 if iso3 == highlight else 2)

    if coordinates is not None and not coordinates.empty:
        islands = coordinates[~coordinates["iso3"].isin(drawn)]
        for _, row in islands.iterrows():
            value = values.get(str(row["iso3"]))
            face = cmap(norm(value)) if value is not None else COLORS["inactive"]
            ax.scatter([row["lon"]], [row["lat"]], s=42, color=face,
                       edgecolors=COLORS["canvas"], linewidths=0.6, zorder=4)
            if str(row["iso3"]) == highlight:
                ax.scatter([row["lon"]], [row["lat"]], s=190, facecolors="none",
                           edgecolors=COLORS["ink"], linewidths=1.4, zorder=5)

    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.margins(0.02)
    _title(ax, title)

    if value_label:
        mappable = ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array([])
        cbar = fig.colorbar(mappable, ax=ax, fraction=0.03, pad=0.01,
                            shrink=0.55)
        cbar.set_label(value_label, fontsize=11, color=COLORS["ink_secondary"])
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(length=0, labelsize=10,
                            colors=COLORS["ink_secondary"])

    fig.tight_layout()
    return fig


def load_africa_geometry(path: Optional[Path] = None) -> Dict[str, object]:
    """Load the committed simplified Africa geometry.

    Args:
        path: Optional explicit JSON path. Defaults to
            ``data/reference/africa_geometry.json`` under the project root.

    Returns:
        Dict with keys ``source`` and ``countries``.

    Raises:
        FileNotFoundError: If the geometry file is missing.
    """
    import json

    from src.theme import project_root

    json_path = Path(path) if path else project_root() / "data/reference/africa_geometry.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Africa geometry not found: {json_path}")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    logger.info("Loaded geometry for %d countries", len(payload.get("countries", [])))
    return payload


def plot_scenario_response(
    labels: Sequence[str],
    effects: Sequence[float],
    ax: Optional[plt.Axes] = None,
    title: str = "Model response to each adjusted indicator",
) -> plt.Figure:
    """Plot the one-at-a-time model response for adjusted indicators.

    Args:
        labels: Indicator display names.
        effects: Change in the prediction (pp) when only that indicator moves.
        ax: Matplotlib axes. If None, creates new figure.
        title: Plot title.

    Returns:
        Matplotlib Figure object.
    """
    if ax is None:
        height = max(2.2, 0.5 * len(labels) + 1.0)
        fig, ax = plt.subplots(figsize=(7.6, height))
    else:
        fig = ax.get_figure()

    order = np.argsort([abs(e) for e in effects])
    sorted_labels = [labels[i] for i in order]
    sorted_effects = [effects[i] for i in order]
    colors = [COLORS["terracotta"] if e >= 0 else COLORS["plum"]
              for e in sorted_effects]

    ax.barh(sorted_labels, sorted_effects, color=colors, height=0.55)
    ax.axvline(0, color=COLORS["border_strong"], linewidth=0.9)
    ax.set_xlabel("Change in predicted growth (pp)")
    _title(ax, title)
    style_axes(ax, grid_axis="x")
    fig.tight_layout()
    return fig


def load_africa_coordinates(
    path: Optional[Path] = None,
) -> pd.DataFrame:
    """Load country centroid coordinates used by the dot map.

    Args:
        path: Optional explicit CSV path. Defaults to
            ``data/reference/africa_centroids.csv`` under the project root.

    Returns:
        DataFrame with columns iso3, country_name, lat, lon.

    Raises:
        FileNotFoundError: If the reference file is missing.
    """
    from src.theme import project_root

    csv_path = Path(path) if path else project_root() / "data/reference/africa_centroids.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Africa centroid reference not found: {csv_path}")
    frame = pd.read_csv(csv_path)
    logger.info("Loaded %d country centroids", len(frame))
    return frame


def coverage_points(
    coordinates: pd.DataFrame,
    values: Dict[str, float],
) -> List[Tuple[float, float, float]]:
    """Build (lon, lat, intensity) triples for the inline SVG dot field.

    Args:
        coordinates: Frame with columns iso3, lat, lon.
        values: iso3 -> intensity in [0, 1]. Missing entries default to 0.

    Returns:
        List of (lon, lat, intensity) triples.
    """
    points: List[Tuple[float, float, float]] = []
    for _, row in coordinates.iterrows():
        points.append((
            float(row["lon"]),
            float(row["lat"]),
            float(values.get(row["iso3"], 0.0)),
        ))
    return points
