"""Editorial UI components for the Africa Growth Explorer.

Every component is a pure function returning an HTML string built from the
design tokens in :mod:`src.theme`, plus a thin ``render`` helper that writes
the markup into a Streamlit container. Keeping the markup pure makes the
layout testable without starting a Streamlit runtime.

No emoji, no icon fonts and no decorative graphics are produced here: the
hierarchy comes from typography, spacing and a single terracotta accent.
"""
from __future__ import annotations

import html as html_lib
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from src.theme import COLORS, MAP_SCALE

__all__ = [
    "render",
    "esc",
    "page_header",
    "hero",
    "africa_map_svg",
    "africa_dot_svg",
    "section",
    "kpi_grid",
    "callout",
    "guardrail",
    "process_strip",
    "numbered_notes",
    "research_table",
    "legend",
    "card",
    "prose",
    "meta",
    "bar_cell",
]


def render(markup: str) -> None:
    """Write raw component markup into the current Streamlit container.

    Args:
        markup: HTML string produced by one of the builders in this module.

    Returns:
        None.
    """
    import streamlit as st

    st.markdown(markup, unsafe_allow_html=True)


def esc(value: object) -> str:
    """HTML-escape any value for safe interpolation.

    Args:
        value: Any object; converted with ``str``.

    Returns:
        Escaped string.
    """
    return html_lib.escape("" if value is None else str(value))


def _tip(text: Optional[str]) -> str:
    """Build a title attribute for progressive disclosure of technical detail.

    Args:
        text: Tooltip text, or None.

    Returns:
        A ``title="..."`` attribute string, or an empty string.
    """
    return f' title="{esc(text)}"' if text else ""


# ---------------------------------------------------------------------------
# Page furniture
# ---------------------------------------------------------------------------

def page_header(eyebrow: str, title: str, standfirst: str = "") -> str:
    """Build the standard page header.

    Args:
        eyebrow: Short uppercase kicker, for example "MODEL PERFORMANCE".
        title: Editorial page title in the display serif.
        standfirst: Optional one or two sentence introduction.

    Returns:
        HTML string.
    """
    lead = (
        f'<p class="ed-standfirst">{esc(standfirst)}</p>' if standfirst else ""
    )
    return (
        '<div class="ed-header">'
        f'<p class="ed-eyebrow">{esc(eyebrow)}</p>'
        f'<h1 class="ed-page-title">{esc(title)}</h1>'
        f"{lead}"
        "</div>"
    )


def hero(eyebrow: str, title: str, lead: str, sub: str, visual: str = "") -> str:
    """Build the Overview hero: editorial statement plus a restrained visual.

    Args:
        eyebrow: Uppercase kicker.
        title: Large display headline.
        lead: Primary sentence describing what the product estimates.
        sub: Secondary sentence with the method and data source.
        visual: Optional inline SVG markup for the right column.

    Returns:
        HTML string.
    """
    right = f'<div class="ed-hero-visual">{visual}</div>' if visual else ""
    return (
        '<div class="ed-hero">'
        "<div>"
        f'<p class="ed-eyebrow">{esc(eyebrow)}</p>'
        f'<h1 class="ed-hero-title">{esc(title)}</h1>'
        f'<p class="ed-hero-lead">{esc(lead)}</p>'
        f'<p class="ed-hero-sub">{esc(sub)}</p>'
        "</div>"
        f"{right}"
        "</div>"
    )


def section(title: str, note: str = "") -> str:
    """Build an editorial section heading with an optional explanatory line.

    Args:
        title: Section title in the display serif.
        note: Optional short sentence under the title.

    Returns:
        HTML string.
    """
    body = f'<p class="ed-section-note">{esc(note)}</p>' if note else ""
    return (
        '<div class="ed-section">'
        f'<h2 class="ed-section-title">{esc(title)}</h2>'
        f"{body}"
        '<hr class="ed-rule" />'
        "</div>"
    )


def prose(text: str) -> str:
    """Wrap a paragraph of editorial copy at reading width.

    Args:
        text: Plain text paragraph.

    Returns:
        HTML string.
    """
    return f'<p class="ed-prose">{esc(text)}</p>'


def meta(text: str) -> str:
    """Render small muted metadata text.

    Args:
        text: Plain text.

    Returns:
        HTML string.
    """
    return f'<p class="ed-meta">{esc(text)}</p>'


def card(title: str, body_html: str, soft: bool = False) -> str:
    """Wrap content in a single card. Use only for meaningful grouping.

    Args:
        title: Card title in the interface font, or an empty string.
        body_html: Pre-built inner HTML.
        soft: Use the soft surface instead of white.

    Returns:
        HTML string.
    """
    klass = "ed-card-soft" if soft else "ed-card"
    heading = f'<p class="ed-card-title">{esc(title)}</p>' if title else ""
    return f'<div class="{klass}">{heading}{body_html}</div>'


# ---------------------------------------------------------------------------
# KPI system
# ---------------------------------------------------------------------------

def kpi_grid(items: Sequence[Dict[str, object]]) -> str:
    """Build a row of key metrics sharing one visual language.

    Each item is a dict with keys:
        label: uppercase metric label (required)
        value: the number as an already formatted string (required)
        note: short explanatory line (optional)
        accent: True to render the value in terracotta (optional, use once)
        tone: "positive" or "negative" for semantic values (optional)
        tooltip: technical definition shown on hover (optional)

    Args:
        items: Sequence of metric dicts.

    Returns:
        HTML string.
    """
    blocks: List[str] = []
    for item in items:
        classes = ["ed-kpi-value"]
        if item.get("accent"):
            classes.append("accent")
        tone = item.get("tone")
        if tone in ("positive", "negative"):
            classes.append(str(tone))
        tooltip = item.get("tooltip")
        label_class = "ed-kpi-label ed-has-tip" if tooltip else "ed-kpi-label"
        note = item.get("note")
        note_html = f'<p class="ed-kpi-note">{esc(note)}</p>' if note else ""
        blocks.append(
            '<div class="ed-kpi">'
            f'<p class="{label_class}"{_tip(tooltip)}>{esc(item["label"])}</p>'
            f'<p class="{" ".join(classes)}">{esc(item["value"])}</p>'
            f"{note_html}"
            "</div>"
        )
    return f'<div class="ed-kpi-grid">{"".join(blocks)}</div>'


# ---------------------------------------------------------------------------
# Callouts, guardrail and notes
# ---------------------------------------------------------------------------

def callout(label: str, paragraphs: Sequence[str], kind: str = "info") -> str:
    """Build an editorial callout panel.

    Args:
        label: Uppercase label, for example "EXTRAPOLATION".
        paragraphs: One or more plain text paragraphs.
        kind: "info", "warning" or "critical".

    Returns:
        HTML string.

    Raises:
        ValueError: If kind is not one of the supported values.
    """
    if kind not in {"info", "warning", "critical"}:
        raise ValueError(f"unsupported callout kind: {kind!r}")
    klass = "ed-callout" if kind == "info" else f"ed-callout {kind}"
    body = "".join(f"<p>{esc(p)}</p>" for p in paragraphs)
    head = f'<p class="ed-callout-label">{esc(label)}</p>' if label else ""
    return f'<div class="{klass}">{head}{body}</div>'


def guardrail(paragraphs: Sequence[str], label: str = "Causal guardrail") -> str:
    """Build the causal guardrail research note.

    Args:
        paragraphs: Plain text paragraphs of the note.
        label: Uppercase label for the note.

    Returns:
        HTML string.
    """
    body = "".join(f"<p>{esc(p)}</p>" for p in paragraphs)
    return (
        '<div class="ed-guardrail">'
        f'<p class="ed-guardrail-label">{esc(label)}</p>'
        f"{body}"
        "</div>"
    )


def process_strip(stages: Sequence[Tuple[str, str]]) -> str:
    """Build the numbered "how it works" process strip.

    Args:
        stages: Sequence of (title, description) pairs, in order.

    Returns:
        HTML string.
    """
    steps: List[str] = []
    for i, (title, body) in enumerate(stages, start=1):
        steps.append(
            '<div class="ed-process-step">'
            f'<div class="ed-process-num">{i:02d}</div>'
            f'<p class="ed-process-title">{esc(title)}</p>'
            f'<p class="ed-process-body">{esc(body)}</p>'
            "</div>"
        )
    return f'<div class="ed-process">{"".join(steps)}</div>'


def numbered_notes(notes: Sequence[Tuple[str, str]]) -> str:
    """Build numbered editorial notes, used for limitations.

    Args:
        notes: Sequence of (title, body) pairs.

    Returns:
        HTML string.
    """
    blocks: List[str] = []
    for i, (title, body) in enumerate(notes, start=1):
        blocks.append(
            '<div class="ed-note">'
            f'<p class="ed-note-num">{i:02d}</p>'
            f'<p class="ed-note-title">{esc(title)}</p>'
            f'<p class="ed-note-body">{esc(body)}</p>'
            "</div>"
        )
    return f'<div class="ed-notes">{"".join(blocks)}</div>'


def legend(items: Sequence[Tuple[str, str]]) -> str:
    """Build a small chart legend using editorial typography.

    Args:
        items: Sequence of (colour hex, label) pairs.

    Returns:
        HTML string.
    """
    parts = [
        '<span class="ed-legend-item">'
        f'<span class="ed-swatch" style="background:{esc(colour)}"></span>'
        f"{esc(label)}</span>"
        for colour, label in items
    ]
    return f'<div class="ed-legend">{"".join(parts)}</div>'


# ---------------------------------------------------------------------------
# Research tables
# ---------------------------------------------------------------------------

def bar_cell(value: float, max_abs: float) -> str:
    """Render a tiny directional bar for a signed model response.

    The bar grows right for positive values and left for negative ones from a
    shared centre line, which reads faster than a column of signed numbers.

    Args:
        value: Signed value to encode.
        max_abs: Largest absolute value in the column, used for scaling.

    Returns:
        HTML string.
    """
    if not math.isfinite(value) or max_abs <= 0:
        width = 0.0
    else:
        width = min(abs(value) / max_abs, 1.0) * 50.0
    colour = COLORS["terracotta"] if value >= 0 else COLORS["plum"]
    offset = 50.0 if value >= 0 else 50.0 - width
    return (
        '<span style="position:relative;display:inline-block;width:72px;'
        'height:10px;vertical-align:middle;">'
        f'<span style="position:absolute;left:50%;top:0;width:1px;height:10px;'
        f'background:{COLORS["border_strong"]};"></span>'
        f'<span style="position:absolute;left:{offset:.1f}%;top:3px;'
        f'width:{width:.1f}%;height:4px;background:{colour};'
        'border-radius:2px;"></span>'
        "</span>"
    )


def research_table(
    df: pd.DataFrame,
    formats: Optional[Dict[str, str]] = None,
    emphasis: Optional[Iterable[str]] = None,
    semantic: Optional[Iterable[str]] = None,
    raw_html: Optional[Iterable[str]] = None,
    highlight_rows: Optional[Iterable[int]] = None,
    scroll: bool = False,
    note: str = "",
) -> str:
    """Render a DataFrame as an editorial research table.

    Numeric columns are right aligned with tabular figures, text columns are
    left aligned, and rows are separated by hairline dividers rather than
    full cell boxes.

    Args:
        df: Data to render. Column labels are used verbatim as headers.
        formats: Optional column -> format spec (for example "{:+.2f}").
        emphasis: Columns rendered in stronger ink.
        semantic: Numeric columns coloured by sign, using muted palette
            positive and negative colours.
        raw_html: Columns whose values are already HTML and must not be
            escaped, for example bar cells.
        highlight_rows: Positional row indices to tint.
        scroll: Constrain height and scroll vertically for long tables.
        note: Optional caption rendered under the table.

    Returns:
        HTML string.
    """
    formats = formats or {}
    emphasis_set = set(emphasis or [])
    semantic_set = set(semantic or [])
    raw_set = set(raw_html or [])
    highlight = set(highlight_rows or [])

    numeric_cols = {
        col for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col]) or col in formats
    } - raw_set

    header_cells = "".join(
        f'<th class="{"num" if col in numeric_cols else ""}">{esc(col)}</th>'
        for col in df.columns
    )

    body_rows: List[str] = []
    for pos, (_, row) in enumerate(df.iterrows()):
        cells: List[str] = []
        for col in df.columns:
            value = row[col]
            classes: List[str] = []
            if col in numeric_cols:
                classes.append("num")
            if col in emphasis_set:
                classes.append("emphasis")
            if col in semantic_set and isinstance(value, (int, float)):
                if pd.notna(value) and value > 0:
                    classes.append("positive")
                elif pd.notna(value) and value < 0:
                    classes.append("negative")

            if col in raw_set:
                text = str(value)
            elif pd.isna(value):
                text = "&ndash;"
            elif col in formats:
                text = esc(formats[col].format(value))
            else:
                text = esc(value)
            cls = f' class="{" ".join(classes)}"' if classes else ""
            cells.append(f"<td{cls}>{text}</td>")
        row_cls = ' class="highlight"' if pos in highlight else ""
        body_rows.append(f"<tr{row_cls}>{''.join(cells)}</tr>")

    wrap_cls = "ed-table-wrap scroll" if scroll else "ed-table-wrap"
    caption = f'<p class="ed-table-note">{esc(note)}</p>' if note else ""
    return (
        f'<div class="{wrap_cls}">'
        '<table class="ed-table">'
        f"<thead><tr>{header_cells}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table></div>"
        f"{caption}"
    )


# ---------------------------------------------------------------------------
# Africa dot field
# ---------------------------------------------------------------------------

def africa_map_svg(
    countries: Sequence[Dict[str, object]],
    values: Optional[Dict[str, float]] = None,
    width: int = 360,
    height: int = 360,
    highlight: Optional[str] = None,
) -> str:
    """Build the Africa motif as an inline SVG map.

    A restrained terracotta outline map: hairline country borders on the warm
    canvas, filled on the terracotta tonal scale where a value is supplied.
    No illustration, no pattern, no stock graphic.

    Args:
        countries: Country records from the committed geometry, each with
            keys ``iso3`` and ``rings`` (lists of [lon, lat] pairs).
        values: Optional iso3 -> intensity in [0, 1] used for the fill tone.
        width: SVG width in pixels.
        height: SVG height in pixels.
        highlight: Optional iso3 stroked in ink.

    Returns:
        SVG markup string. Empty string when no geometry is supplied.
    """
    if not countries:
        return ""
    values = values or {}

    xs: List[float] = []
    ys: List[float] = []
    for country in countries:
        for ring in country["rings"]:  # type: ignore[index]
            for lon, lat in ring:
                xs.append(float(lon))
                ys.append(float(lat))
    if not xs:
        return ""

    lon_min, lon_max = min(xs), max(xs)
    lat_min, lat_max = min(ys), max(ys)
    span_x = max(lon_max - lon_min, 1e-6)
    span_y = max(lat_max - lat_min, 1e-6)
    scale = min(width / span_x, height / span_y)
    off_x = (width - span_x * scale) / 2.0
    off_y = (height - span_y * scale) / 2.0

    present = [v for v in values.values() if v is not None]
    lo = min(present) if present else 0.0
    hi = max(present) if present else 1.0
    span_v = (hi - lo) or 1.0

    def shade(iso3: str) -> str:
        value = values.get(iso3)
        if value is None:
            return COLORS["inactive"]
        idx = int(min(max((value - lo) / span_v, 0.0), 0.999) * len(MAP_SCALE))
        return MAP_SCALE[idx]

    paths: List[str] = []
    for country in countries:
        iso3 = str(country["iso3"])
        commands: List[str] = []
        for ring in country["rings"]:  # type: ignore[index]
            points = [
                (
                    off_x + (float(lon) - lon_min) * scale,
                    off_y + (lat_max - float(lat)) * scale,
                )
                for lon, lat in ring
            ]
            head = f"M{points[0][0]:.1f},{points[0][1]:.1f}"
            tail = "".join(f"L{x:.1f},{y:.1f}" for x, y in points[1:])
            commands.append(f"{head}{tail}Z")
        stroke = COLORS["ink"] if iso3 == highlight else COLORS["canvas"]
        stroke_width = 1.2 if iso3 == highlight else 0.6
        paths.append(
            f'<path d="{"".join(commands)}" fill="{shade(iso3)}" '
            f'stroke="{stroke}" stroke-width="{stroke_width}" '
            'stroke-linejoin="round" />'
        )

    return (
        f'<svg viewBox="0 0 {width} {height}" width="{width}" '
        f'height="{height}" role="img" aria-label="Map of Africa" '
        'xmlns="http://www.w3.org/2000/svg">'
        f"{''.join(paths)}"
        "</svg>"
    )


def africa_dot_svg(
    points: Sequence[Tuple[float, float, float]],
    width: int = 360,
    height: int = 340,
    highlight: Optional[Tuple[float, float]] = None,
    radius: float = 5.0,
) -> str:
    """Build the restrained Africa motif as an inline SVG dot field.

    Each country is one dot placed at its centroid on an equirectangular
    projection, shaded on the terracotta tonal scale by the supplied
    intensity. No outline, no illustration, no stock graphic.

    Args:
        points: Sequence of (longitude, latitude, intensity) triples where
            intensity is in [0, 1].
        width: SVG width in pixels.
        height: SVG height in pixels.
        highlight: Optional (longitude, latitude) drawn with a deep terracotta
            ring, used to mark the selected country.
        radius: Dot radius in pixels.

    Returns:
        SVG markup string. Empty string when no points are supplied.
    """
    if not points:
        return ""

    lons = [p[0] for p in points]
    lats = [p[1] for p in points]
    lon_min, lon_max = min(lons), max(lons)
    lat_min, lat_max = min(lats), max(lats)
    pad = 18.0
    span_x = max(lon_max - lon_min, 1e-6)
    span_y = max(lat_max - lat_min, 1e-6)

    def project(lon: float, lat: float) -> Tuple[float, float]:
        x = pad + (lon - lon_min) / span_x * (width - 2 * pad)
        y = pad + (lat_max - lat) / span_y * (height - 2 * pad)
        return x, y

    def shade(intensity: float) -> str:
        if not math.isfinite(intensity):
            return COLORS["inactive"]
        idx = int(min(max(intensity, 0.0), 0.999) * len(MAP_SCALE))
        return MAP_SCALE[idx]

    circles: List[str] = []
    for lon, lat, intensity in points:
        x, y = project(lon, lat)
        circles.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" '
            f'fill="{shade(intensity)}" />'
        )

    if highlight is not None:
        hx, hy = project(*highlight)
        circles.append(
            f'<circle cx="{hx:.1f}" cy="{hy:.1f}" r="{radius + 4:.1f}" '
            f'fill="none" stroke="{COLORS["terracotta_deep"]}" '
            'stroke-width="1.5" />'
        )

    return (
        f'<svg viewBox="0 0 {width} {height}" width="{width}" '
        f'height="{height}" role="img" aria-label="Africa data coverage" '
        'xmlns="http://www.w3.org/2000/svg">'
        f"{''.join(circles)}"
        "</svg>"
    )
