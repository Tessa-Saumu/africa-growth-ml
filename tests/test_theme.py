"""Tests for the Terracotta Editorial design tokens and stylesheet.

These are contract tests for the design system: the token set, the font
registration path used by server-rendered charts, and the rule that no colour
may appear in the stylesheet unless it is a declared token.
"""
import re
from pathlib import Path

import pytest

from src import theme

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_colors_expose_the_documented_tokens():
    """Every token named in the design specification must exist."""
    required = {
        "canvas", "surface", "surface_soft", "ink", "ink_secondary", "muted",
        "border", "border_strong", "terracotta", "terracotta_deep",
        "terracotta_light", "terracotta_tint", "plum", "rose", "sand",
        "positive", "warning", "critical",
    }
    assert required <= set(theme.COLORS)
    assert theme.COLORS["canvas"] == "#FCFAF7"
    assert theme.COLORS["terracotta"] == "#C65A35"
    assert theme.COLORS["ink"] == "#241D19"


def test_get_colors_returns_a_copy():
    """Callers must not be able to mutate the shared token table."""
    colors = theme.get_colors()
    colors["canvas"] = "#000000"
    assert theme.COLORS["canvas"] == "#FCFAF7"


def test_chart_sequence_is_the_fixed_five():
    """The categorical sequence is fixed and terracotta leads it."""
    assert theme.get_chart_colors() == [
        "#C65A35", "#59404A", "#B98278", "#D9B88C", "#72745F",
    ]


def test_no_bright_default_colors_in_tokens():
    """No neon or Bootstrap-style colours may enter the palette."""
    forbidden = {"#00FF00", "#FF0000", "#0000FF", "#007BFF", "#1F77B4"}
    assert not (forbidden & {v.upper() for v in theme.COLORS.values()})


def test_css_only_uses_declared_tokens():
    """Every hex colour in the stylesheet must be a design token.

    White is allowed because it is the on-terracotta text colour.
    """
    css = theme.build_editorial_css()
    used = {m.upper() for m in re.findall(r"#[0-9A-Fa-f]{6}", css)}
    allowed = {v.upper() for v in theme.COLORS.values()} | {"#FFFFFF"}
    assert used <= allowed, f"undeclared colours in CSS: {sorted(used - allowed)}"


def test_css_has_no_unbalanced_format_braces():
    """The stylesheet is built from an f-string; literal braces must survive."""
    css = theme.build_editorial_css()
    assert "{{" not in css and "}}" not in css
    assert css.count("{") == css.count("}")


def test_css_declares_the_editorial_fonts_and_canvas():
    css = theme.build_editorial_css()
    assert "Instrument Serif" in css
    assert "DM Sans" in css
    assert theme.COLORS["canvas"] in css
    assert "ed-page-title" in css and "ed-kpi-value" in css


def test_css_contains_responsive_rules():
    """Layouts must reflow rather than only shrink."""
    css = theme.build_editorial_css()
    assert css.count("@media") >= 2
    assert "grid-template-columns: 1fr" in css


def test_bundled_fonts_are_present_in_the_repository():
    """Charts are rendered server side, so the fonts must ship with the app."""
    paths = theme.font_paths(REPO_ROOT)
    names = {p.name for p in paths}
    assert "DMSans-Regular.ttf" in names
    assert "InstrumentSerif-Regular.ttf" in names


def test_register_project_fonts_exposes_families_to_matplotlib():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    families = theme.register_project_fonts(REPO_ROOT)
    assert "DM Sans" in families
    assert "Instrument Serif" in families

    from matplotlib import font_manager
    resolved = font_manager.findfont(
        font_manager.FontProperties(family="DM Sans"), fallback_to_default=False
    )
    assert resolved.endswith("DMSans-Regular.ttf")


def test_project_root_resolves_to_the_repository():
    assert (theme.project_root() / "pyproject.toml").exists()
