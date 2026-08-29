"""Terracotta Editorial design system for the Africa Growth Explorer.

Single source of truth for the application's visual language:

- ``COLORS``: the design tokens (canvas, ink, terracotta, semantic colours).
- ``CHART_COLORS``: the fixed chart colour sequence.
- ``register_project_fonts``: makes the bundled Instrument Serif / DM Sans
  files available to Matplotlib so server-rendered charts use the same
  typefaces as the browser UI.
- ``build_editorial_css`` / ``inject_editorial_styles``: one centralised CSS
  injection for the whole Streamlit application.

No other module may introduce colours, fonts or spacing values of its own.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------

COLORS: Dict[str, str] = {
    # Base surfaces
    "canvas": "#FCFAF7",
    "surface": "#FFFFFF",
    "surface_soft": "#F7F0EA",
    # Ink
    "ink": "#241D19",
    "ink_secondary": "#5E524B",
    "muted": "#8B7D74",
    # Borders
    "border": "#E7DDD5",
    "border_strong": "#D7C9BF",
    # Terracotta accents
    "terracotta": "#C65A35",
    "terracotta_deep": "#A94325",
    "terracotta_light": "#E8B39C",
    "terracotta_tint": "#F4E4DA",
    # Supporting analytical accents
    "plum": "#59404A",
    "rose": "#B98278",
    "sand": "#D9B88C",
    "olive": "#72745F",
    # Semantic
    "positive": "#637A5A",
    "negative": "#9B4637",
    "warning": "#B57A32",
    "critical": "#8C332B",
    # Callout backgrounds
    "warning_bg": "#FAF1E2",
    "critical_bg": "#F5E3DF",
    # Inactive / non-significant data
    "inactive": "#D9D0CA",
}

# Fixed chart colour sequence (spec section 53). Prefer one or two per chart.
CHART_COLORS: List[str] = [
    COLORS["terracotta"],
    COLORS["plum"],
    COLORS["rose"],
    COLORS["sand"],
    COLORS["olive"],
]

# Terracotta tonal scale, light to deep (spec section 25).
MAP_SCALE: List[str] = [
    COLORS["terracotta_tint"],
    COLORS["terracotta_light"],
    COLORS["terracotta"],
    COLORS["terracotta_deep"],
]

FONT_DISPLAY = "Instrument Serif"
FONT_UI = "DM Sans"
FONT_DISPLAY_STACK = '"Instrument Serif", Georgia, "Times New Roman", serif'
FONT_UI_STACK = '"DM Sans", Arial, sans-serif'

# 8px base spacing scale (spec section 48).
SPACING: Dict[str, str] = {
    "micro": "4px",
    "xs": "8px",
    "sm": "12px",
    "md": "16px",
    "lg": "24px",
    "xl": "32px",
    "2xl": "48px",
    "3xl": "64px",
}

SHADOW_DEFAULT = "0 2px 10px rgba(36, 29, 25, 0.04)"
SHADOW_ELEVATED = "0 6px 24px rgba(36, 29, 25, 0.07)"

CONTENT_MAX_WIDTH = "1400px"
PROSE_MAX_WIDTH = "780px"

# Bundled OFL fonts, served to the browser by Streamlit static file serving
# and read directly by Matplotlib for chart typography.
FONT_FILES: Dict[str, str] = {
    "DM Sans Regular": "static/fonts/DMSans-Regular.ttf",
    "DM Sans Medium": "static/fonts/DMSans-Medium.ttf",
    "DM Sans SemiBold": "static/fonts/DMSans-SemiBold.ttf",
    "Instrument Serif Regular": "static/fonts/InstrumentSerif-Regular.ttf",
}


def get_colors() -> Dict[str, str]:
    """Return a copy of the design tokens.

    Returns:
        Dict mapping token name to hex colour string.
    """
    return COLORS.copy()


def get_chart_colors() -> List[str]:
    """Return a copy of the fixed chart colour sequence.

    Returns:
        List of hex colour strings, primary series first.
    """
    return list(CHART_COLORS)


def project_root(start: Optional[Path] = None) -> Path:
    """Resolve the repository root by walking up to the pyproject.toml.

    Args:
        start: Directory to start from. Defaults to this file's location.

    Returns:
        Path of the first ancestor containing pyproject.toml, else ``start``.
    """
    here = Path(start) if start is not None else Path(__file__).resolve().parent
    for candidate in [here, *here.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return here


def font_paths(root: Optional[Path] = None) -> List[Path]:
    """Absolute paths of the bundled font files that exist on disk.

    Args:
        root: Repository root. Resolved automatically when omitted.

    Returns:
        List of existing .ttf paths (possibly empty).
    """
    base = project_root() if root is None else Path(root)
    return [base / rel for rel in FONT_FILES.values() if (base / rel).exists()]


def register_project_fonts(root: Optional[Path] = None) -> List[str]:
    """Register the bundled fonts with Matplotlib's font manager.

    Charts are rendered server side, so the typefaces must be loaded from the
    repository rather than assumed present on the host.

    Args:
        root: Repository root. Resolved automatically when omitted.

    Returns:
        Sorted list of font family names now available to Matplotlib. Empty
        when no bundled font files were found.
    """
    from matplotlib import font_manager

    registered: List[str] = []
    for path in font_paths(root):
        try:
            font_manager.fontManager.addfont(str(path))
            registered.append(
                font_manager.FontProperties(fname=str(path)).get_name()
            )
        except (RuntimeError, OSError) as exc:  # pragma: no cover - host issue
            logger.warning("Could not register font %s: %s", path, exc)
    families = sorted(set(registered))
    if families:
        logger.info("Registered project fonts: %s", ", ".join(families))
    else:
        logger.warning(
            "No bundled fonts found under static/fonts; charts fall back to "
            "the Matplotlib default sans-serif face"
        )
    return families


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

def build_editorial_css() -> str:
    """Build the single stylesheet for the application.

    Kept pure (returns a string, renders nothing) so it can be inspected in
    tests. All values come from the design tokens above.

    Returns:
        A CSS string, without the surrounding <style> tag.
    """
    c = COLORS
    return f"""
:root {{
    --canvas: {c['canvas']};
    --surface: {c['surface']};
    --surface-soft: {c['surface_soft']};
    --ink: {c['ink']};
    --ink-secondary: {c['ink_secondary']};
    --muted: {c['muted']};
    --border: {c['border']};
    --border-strong: {c['border_strong']};
    --terracotta: {c['terracotta']};
    --terracotta-deep: {c['terracotta_deep']};
    --terracotta-light: {c['terracotta_light']};
    --terracotta-tint: {c['terracotta_tint']};
    --plum: {c['plum']};
    --positive: {c['positive']};
    --negative: {c['negative']};
    --warning: {c['warning']};
    --critical: {c['critical']};
    --font-display: {FONT_DISPLAY_STACK};
    --font-ui: {FONT_UI_STACK};
    --shadow-default: {SHADOW_DEFAULT};
    --shadow-elevated: {SHADOW_ELEVATED};
}}

/* ---------------------------------------------------------------- canvas */
html, body, .stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {{
    background: var(--canvas) !important;
    color: var(--ink);
    font-family: var(--font-ui);
}}

[data-testid="stHeader"],
[data-testid="stToolbar"] {{
    background: transparent;
    box-shadow: none;
}}

[data-testid="stDecoration"] {{ display: none; }}

[data-testid="stMainBlockContainer"] {{
    max-width: {CONTENT_MAX_WIDTH} !important;
    padding: 48px 40px 96px 40px !important;
}}

/* ------------------------------------------------------------ typography */
[data-testid="stMain"] p,
[data-testid="stMain"] li,
[data-testid="stMain"] label,
[data-testid="stMain"] div {{
    font-family: var(--font-ui);
}}

[data-testid="stMain"] [data-testid="stMarkdownContainer"] p {{
    font-size: 15px;
    line-height: 1.6;
    color: var(--ink-secondary);
    max-width: {PROSE_MAX_WIDTH};
}}

[data-testid="stMain"] [data-testid="stMarkdownContainer"] li {{
    font-size: 15px;
    line-height: 1.6;
    color: var(--ink-secondary);
}}

[data-testid="stHeading"] h1,
[data-testid="stHeading"] h2,
[data-testid="stHeading"] h3 {{
    font-family: var(--font-display);
    font-weight: 400;
    color: var(--ink);
    letter-spacing: -0.01em;
}}

a, a:visited {{ color: var(--terracotta-deep); }}
a:hover {{ color: var(--terracotta); }}

/* Editorial building blocks -------------------------------------------- */
.ed-eyebrow {{
    font-family: var(--font-ui);
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--terracotta);
    margin: 0 0 12px 0;
}}

.ed-page-title {{
    font-family: var(--font-display);
    font-size: 42px;
    line-height: 1.05;
    font-weight: 400;
    letter-spacing: -0.02em;
    color: var(--ink);
    margin: 0 0 16px 0;
}}

.ed-standfirst {{
    font-family: var(--font-ui);
    font-size: 16px;
    line-height: 1.6;
    color: var(--ink-secondary);
    max-width: {PROSE_MAX_WIDTH};
    margin: 0;
}}

.ed-header {{
    margin: 0 0 8px 0;
    padding-bottom: 24px;
    border-bottom: 1px solid var(--border);
}}

.ed-section {{
    margin: 40px 0 16px 0;
}}

.ed-section-title {{
    font-family: var(--font-display);
    font-size: 28px;
    line-height: 1.15;
    font-weight: 400;
    color: var(--ink);
    margin: 0;
}}

.ed-section-note {{
    font-family: var(--font-ui);
    font-size: 15px;
    line-height: 1.6;
    color: var(--ink-secondary);
    max-width: {PROSE_MAX_WIDTH};
    margin: 8px 0 0 0;
}}

.ed-rule {{
    height: 1px;
    background: var(--border);
    border: 0;
    margin: 12px 0 0 0;
}}

.ed-prose {{
    font-family: var(--font-ui);
    font-size: 15px;
    line-height: 1.6;
    color: var(--ink-secondary);
    max-width: {PROSE_MAX_WIDTH};
}}

.ed-meta {{
    font-family: var(--font-ui);
    font-size: 12px;
    line-height: 1.4;
    font-weight: 500;
    color: var(--muted);
}}

/* KPI ------------------------------------------------------------------- */
.ed-kpi-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 8px 32px;
    margin: 8px 0 0 0;
}}

.ed-kpi {{
    padding: 20px 0 4px 0;
    border-top: 1px solid var(--border);
}}

.ed-kpi-label {{
    font-family: var(--font-ui);
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin: 0 0 10px 0;
}}

.ed-kpi-value {{
    font-family: var(--font-display);
    font-size: 36px;
    line-height: 1;
    font-weight: 400;
    color: var(--ink);
    margin: 0;
}}

.ed-kpi-value.accent {{ color: var(--terracotta); }}
.ed-kpi-value.positive {{ color: var(--positive); }}
.ed-kpi-value.negative {{ color: var(--negative); }}

.ed-kpi-note {{
    font-family: var(--font-ui);
    font-size: 12px;
    line-height: 1.45;
    color: var(--muted);
    margin: 10px 0 0 0;
    max-width: 34ch;
}}

.ed-has-tip {{
    border-bottom: 1px dotted var(--border-strong);
    cursor: help;
}}

/* Cards ----------------------------------------------------------------- */
.ed-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px;
    box-shadow: var(--shadow-default);
}}

.ed-card-soft {{
    background: var(--surface-soft);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px;
}}

.ed-card-title {{
    font-family: var(--font-ui);
    font-size: 14px;
    font-weight: 600;
    color: var(--ink);
    margin: 0 0 8px 0;
}}

/* Hero ------------------------------------------------------------------ */
.ed-hero {{
    display: grid;
    grid-template-columns: minmax(0, 1.35fr) minmax(0, 1fr);
    gap: 48px;
    align-items: center;
    padding: 8px 0 32px 0;
    border-bottom: 1px solid var(--border);
}}

.ed-hero-title {{
    font-family: var(--font-display);
    font-size: 52px;
    line-height: 1.02;
    letter-spacing: -0.02em;
    font-weight: 400;
    color: var(--ink);
    margin: 0 0 20px 0;
}}

.ed-hero-lead {{
    font-family: var(--font-ui);
    font-size: 17px;
    line-height: 1.55;
    color: var(--ink-secondary);
    margin: 0 0 12px 0;
    max-width: 46ch;
}}

.ed-hero-sub {{
    font-family: var(--font-ui);
    font-size: 14px;
    line-height: 1.6;
    color: var(--muted);
    margin: 0;
    max-width: 46ch;
}}

.ed-hero-visual {{
    display: flex;
    justify-content: flex-end;
}}

.ed-hero-visual svg {{ width: 100%; max-width: 360px; height: auto; }}

/* Process strip --------------------------------------------------------- */
.ed-process {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 0;
    margin: 8px 0 0 0;
    border-top: 1px solid var(--border);
}}

.ed-process-step {{
    position: relative;
    padding: 24px 24px 8px 0;
}}

.ed-process-step::before {{
    content: "";
    position: absolute;
    top: -1px;
    left: 0;
    width: 100%;
    height: 1px;
    background: var(--border);
}}

.ed-process-num {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 30px;
    height: 30px;
    border-radius: 50%;
    border: 1px solid var(--terracotta);
    background: var(--terracotta-tint);
    color: var(--terracotta-deep);
    font-family: var(--font-ui);
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.04em;
    margin-bottom: 14px;
}}

.ed-process-title {{
    font-family: var(--font-ui);
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--ink);
    margin: 0 0 6px 0;
}}

.ed-process-body {{
    font-family: var(--font-ui);
    font-size: 13px;
    line-height: 1.5;
    color: var(--ink-secondary);
    margin: 0;
    max-width: 24ch;
}}

/* Callouts and the causal guardrail ------------------------------------- */
.ed-callout {{
    border-radius: 12px;
    padding: 20px 24px;
    margin: 8px 0;
    border: 1px solid var(--border);
    background: var(--surface-soft);
    border-left: 3px solid var(--terracotta);
}}

.ed-callout.warning {{
    background: {c['warning_bg']};
    border-left-color: var(--warning);
}}

.ed-callout.critical {{
    background: {c['critical_bg']};
    border-left-color: var(--critical);
}}

.ed-callout-label {{
    font-family: var(--font-ui);
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--terracotta-deep);
    margin: 0 0 8px 0;
}}

.ed-callout.warning .ed-callout-label {{ color: {c['warning']}; }}
.ed-callout.critical .ed-callout-label {{ color: {c['critical']}; }}

.ed-callout p {{
    font-family: var(--font-ui);
    font-size: 14px;
    line-height: 1.6;
    color: var(--ink-secondary);
    margin: 0 0 10px 0;
    max-width: 72ch;
}}

.ed-callout p:last-child {{ margin-bottom: 0; }}

.ed-guardrail {{
    border-left: 2px solid var(--terracotta);
    padding: 4px 0 4px 24px;
    margin: 8px 0;
}}

.ed-guardrail-label {{
    font-family: var(--font-ui);
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--terracotta);
    margin: 0 0 10px 0;
}}

.ed-guardrail p {{
    font-family: var(--font-ui);
    font-size: 14px;
    line-height: 1.65;
    color: var(--ink-secondary);
    margin: 0 0 10px 0;
    max-width: 68ch;
}}

.ed-guardrail p:last-child {{ margin-bottom: 0; }}

/* Numbered editorial notes ---------------------------------------------- */
.ed-notes {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 32px;
    margin-top: 8px;
}}

.ed-note {{
    border-top: 1px solid var(--border);
    padding-top: 16px;
}}

.ed-note-num {{
    font-family: var(--font-display);
    font-size: 20px;
    color: var(--terracotta);
    margin: 0 0 6px 0;
}}

.ed-note-title {{
    font-family: var(--font-ui);
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--ink);
    margin: 0 0 8px 0;
}}

.ed-note-body {{
    font-family: var(--font-ui);
    font-size: 14px;
    line-height: 1.6;
    color: var(--ink-secondary);
    margin: 0;
}}

/* Research tables -------------------------------------------------------- */
.ed-table-wrap {{
    overflow-x: auto;
    border-top: 1px solid var(--border-strong);
    border-bottom: 1px solid var(--border-strong);
}}

.ed-table-wrap.scroll {{ max-height: 420px; overflow-y: auto; }}

table.ed-table {{
    width: 100%;
    border-collapse: collapse;
    font-family: var(--font-ui);
    font-size: 13px;
    color: var(--ink);
    background: transparent;
}}

table.ed-table thead th {{
    position: sticky;
    top: 0;
    background: var(--surface-soft);
    color: var(--ink-secondary);
    font-weight: 600;
    font-size: 11px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    text-align: left;
    padding: 12px 16px;
    border-bottom: 1px solid var(--border-strong);
    white-space: nowrap;
}}

table.ed-table tbody td {{
    padding: 11px 16px;
    border-bottom: 1px solid var(--border);
    vertical-align: top;
    line-height: 1.45;
}}

table.ed-table tbody tr:last-child td {{ border-bottom: none; }}
table.ed-table tbody tr:hover td {{ background: var(--terracotta-tint); }}

table.ed-table th.num, table.ed-table td.num {{
    text-align: right;
    font-variant-numeric: tabular-nums;
}}

table.ed-table td.emphasis {{
    color: var(--ink);
    font-weight: 600;
}}

table.ed-table td.positive {{ color: var(--positive); }}
table.ed-table td.negative {{ color: var(--negative); }}
table.ed-table tr.highlight td {{ background: var(--terracotta-tint); }}

.ed-table-note {{
    font-family: var(--font-ui);
    font-size: 12px;
    line-height: 1.5;
    color: var(--muted);
    margin: 10px 0 0 0;
    max-width: {PROSE_MAX_WIDTH};
}}

/* Legend ----------------------------------------------------------------- */
.ed-legend {{
    display: flex;
    flex-wrap: wrap;
    gap: 20px;
    margin: 12px 0 0 0;
}}

.ed-legend-item {{
    display: flex;
    align-items: center;
    gap: 8px;
    font-family: var(--font-ui);
    font-size: 12px;
    color: var(--ink-secondary);
}}

.ed-swatch {{
    width: 10px;
    height: 10px;
    border-radius: 50%;
    display: inline-block;
}}

/* ---------------------------------------------------------------- sidebar */
[data-testid="stSidebar"] {{
    background: var(--surface-soft) !important;
    border-right: 1px solid var(--border) !important;
    width: 260px !important;
    min-width: 260px !important;
}}

[data-testid="stSidebarContent"] {{ background: transparent; }}

[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {{
    padding: 8px 20px 32px 20px;
}}

[data-testid="stSidebarHeader"] {{ padding-bottom: 0; }}

.ed-brand {{ margin: 0 0 28px 0; }}

.ed-brand-top {{
    font-family: var(--font-ui);
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--terracotta);
    margin: 0;
}}

.ed-brand-main {{
    font-family: var(--font-display);
    font-size: 20px;
    line-height: 1.1;
    color: var(--ink);
    margin: 2px 0 0 0;
}}

.ed-rail-label {{
    font-family: var(--font-ui);
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--muted);
    margin: 0 0 8px 0;
}}

/* Navigation rail built from sidebar buttons: active is a filled terracotta
   row, inactive is quiet text, hover is a light terracotta tint. */
[data-testid="stSidebar"] .stButton > button {{
    width: 100%;
    justify-content: flex-start;
    text-align: left;
    padding: 9px 12px;
    margin: 0 0 2px 0;
    border-radius: 8px;
    font-family: var(--font-ui);
    font-size: 14px;
    font-weight: 500;
    min-height: 0;
}}

[data-testid="stSidebar"] [data-testid="stBaseButton-tertiary"] {{
    background: transparent !important;
    border: 1px solid transparent !important;
    color: var(--ink-secondary) !important;
}}

[data-testid="stSidebar"] [data-testid="stBaseButton-tertiary"]:hover {{
    background: var(--terracotta-tint) !important;
    color: var(--terracotta-deep) !important;
    border-color: transparent !important;
}}

[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {{
    background: var(--terracotta) !important;
    border: 1px solid var(--terracotta) !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
}}

[data-testid="stSidebar"] [data-testid="stBaseButton-primary"]:hover {{
    background: var(--terracotta-deep);
    border-color: var(--terracotta-deep);
}}

[data-testid="stSidebar"] hr {{
    border-color: var(--border);
    margin: 20px 0;
}}

[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {{
    font-family: var(--font-ui);
    font-size: 10px !important;
    font-weight: 600 !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
}}

/* ----------------------------------------------------------------- inputs */
[data-testid="stMain"] [data-testid="stWidgetLabel"] p {{
    font-family: var(--font-ui);
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: 0;
    color: var(--ink) !important;
}}

div[data-baseweb="select"] > div {{
    background: var(--surface) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: 8px;
    font-family: var(--font-ui);
    font-size: 14px;
    color: var(--ink);
    box-shadow: none;
    transition: border-color 120ms ease;
}}

div[data-baseweb="select"] > div:hover {{ border-color: var(--terracotta-light); }}

div[data-baseweb="select"] > div:focus-within {{
    border-color: var(--terracotta) !important;
    box-shadow: none !important;
}}

div[data-baseweb="popover"] li {{ font-family: var(--font-ui); font-size: 14px; }}

div[data-baseweb="popover"] li[aria-selected="true"] {{
    background: var(--terracotta-tint);
    color: var(--ink);
}}

/* Multiselect chips: small, square-ish, terracotta tint (no loud pills) */
span[data-baseweb="tag"] {{
    background: var(--terracotta-tint) !important;
    color: var(--terracotta-deep) !important;
    border-radius: 6px !important;
    font-family: var(--font-ui);
    font-size: 12px;
}}

span[data-baseweb="tag"] svg {{ fill: var(--terracotta-deep); }}

/* Sliders */
[data-testid="stSliderThumbValue"] {{
    font-family: var(--font-ui);
    font-size: 12px;
    font-weight: 600;
    color: var(--ink);
}}

[data-testid="stSliderTickBar"],
[data-testid="stSliderTickBar"] p {{
    font-family: var(--font-ui);
    font-size: 11px;
    color: var(--muted);
}}

/* Buttons */
[data-testid="stBaseButton-primary"] {{
    background: var(--terracotta);
    border: 1px solid var(--terracotta);
    color: #FFFFFF;
    border-radius: 8px;
    font-family: var(--font-ui);
    font-weight: 600;
    font-size: 14px;
}}

[data-testid="stBaseButton-primary"]:hover {{
    background: var(--terracotta-deep);
    border-color: var(--terracotta-deep);
    color: #FFFFFF;
}}

[data-testid="stBaseButton-secondary"] {{
    background: transparent;
    border: 1px solid var(--terracotta);
    color: var(--terracotta-deep);
    border-radius: 8px;
    font-family: var(--font-ui);
    font-weight: 500;
    font-size: 14px;
}}

[data-testid="stBaseButton-secondary"]:hover {{
    background: var(--terracotta-tint);
    border-color: var(--terracotta-deep);
    color: var(--terracotta-deep);
}}

/* Expanders as quiet disclosure rows */
[data-testid="stExpander"] details {{
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    background: var(--surface) !important;
    box-shadow: none !important;
}}

[data-testid="stExpander"] summary {{
    font-family: var(--font-ui);
    font-size: 13px;
    font-weight: 500;
    color: var(--ink-secondary);
    padding: 12px 16px;
}}

[data-testid="stExpander"] summary:hover {{ color: var(--terracotta-deep); }}

[data-testid="stExpanderDetails"] {{ padding: 4px 16px 16px 16px; }}

/* Alerts: only used as a fallback; keep them editorial rather than neon */
[data-testid="stAlert"] {{
    border-radius: 12px !important;
    border: 1px solid var(--border) !important;
    border-left: 3px solid var(--terracotta) !important;
    background: var(--surface-soft) !important;
    color: var(--ink-secondary) !important;
    font-family: var(--font-ui);
}}

/* Captions */
[data-testid="stCaptionContainer"] p {{
    font-family: var(--font-ui);
    font-size: 12px;
    line-height: 1.5;
    color: var(--muted);
}}

/* Charts sit directly on the canvas */
[data-testid="stImage"] img, [data-testid="stImageContainer"] img {{
    background: transparent;
}}

/* Tabs and focus rings must not fall back to Streamlit blue */
:focus-visible {{ outline: 2px solid var(--terracotta) !important; outline-offset: 2px; }}

/* ------------------------------------------------------------- responsive */
@media (max-width: 1100px) {{
    .ed-hero {{ grid-template-columns: 1fr; gap: 24px; }}
    .ed-hero-visual {{ justify-content: flex-start; }}
    .ed-hero-title {{ font-size: 42px; }}
}}

@media (max-width: 800px) {{
    [data-testid="stMainBlockContainer"] {{ padding: 24px 16px 64px 16px; }}
    .ed-page-title {{ font-size: 32px; }}
    .ed-hero-title {{ font-size: 34px; }}
    .ed-section-title {{ font-size: 23px; }}
    .ed-kpi-value {{ font-size: 30px; }}
    .ed-kpi-grid {{ grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 0 20px; }}
    .ed-notes {{ grid-template-columns: 1fr; gap: 20px; }}
}}
"""


def inject_editorial_styles() -> None:
    """Inject the design system stylesheet into the running Streamlit app.

    Call once, at the top of the application, before any content is rendered.

    Returns:
        None.
    """
    import streamlit as st

    st.markdown(f"<style>{build_editorial_css()}</style>", unsafe_allow_html=True)
