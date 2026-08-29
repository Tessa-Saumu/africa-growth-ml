"""Tests for the editorial UI components.

The builders are pure functions returning HTML, so the page composition can
be checked without starting a Streamlit runtime: escaping, table alignment,
semantic colouring, and the restrained Africa graphics.
"""
import pandas as pd
import pytest

from src import ui
from src.theme import COLORS

EMOJI_RANGES = (
    (0x1F300, 0x1FAFF),
    (0x2600, 0x27BF),
    (0x1F000, 0x1F2FF),
    (0xFE0F, 0xFE0F),
)


def _has_emoji(text: str) -> bool:
    """Return True when the string contains an emoji code point."""
    return any(lo <= ord(ch) <= hi for ch in text for lo, hi in EMOJI_RANGES)


# ----------------------------------------------------------------------------
# Page furniture
# ----------------------------------------------------------------------------

def test_page_header_uses_editorial_hierarchy():
    html = ui.page_header("Model performance", "What the model gets right",
                          "One sentence of context.")
    assert 'class="ed-eyebrow"' in html
    assert 'class="ed-page-title"' in html
    assert "One sentence of context." in html
    assert not _has_emoji(html)


def test_page_header_omits_empty_standfirst():
    assert "ed-standfirst" not in ui.page_header("A", "B")


def test_section_renders_title_note_and_rule():
    html = ui.section("Baseline comparison", "Short explanation.")
    assert 'class="ed-section-title"' in html
    assert "Short explanation." in html
    assert 'class="ed-rule"' in html


def test_components_escape_user_content():
    html = ui.page_header("x", "<script>alert(1)</script>", "&")
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


# ----------------------------------------------------------------------------
# KPI system
# ----------------------------------------------------------------------------

def test_kpi_grid_marks_only_the_requested_accent():
    html = ui.kpi_grid([
        {"label": "Test MAE", "value": "1.82 pp", "note": "n", "accent": True},
        {"label": "Test RMSE", "value": "2.79 pp"},
    ])
    assert html.count("ed-kpi-value accent") == 1
    assert html.count('class="ed-kpi"') == 2


def test_kpi_grid_supports_semantic_tone_and_tooltip():
    html = ui.kpi_grid([
        {"label": "Change", "value": "+0.67 pp", "tone": "positive",
         "tooltip": "Scenario minus baseline."},
    ])
    assert "ed-kpi-value positive" in html
    assert 'title="Scenario minus baseline."' in html
    assert "ed-has-tip" in html


# ----------------------------------------------------------------------------
# Callouts, guardrail, notes
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["info", "warning", "critical"])
def test_callout_kinds(kind):
    html = ui.callout("Extrapolation", ["Body copy."], kind=kind)
    assert "ed-callout" in html
    assert "Body copy." in html
    if kind != "info":
        assert kind in html


def test_callout_rejects_unknown_kind():
    with pytest.raises(ValueError):
        ui.callout("x", ["y"], kind="danger")


def test_guardrail_keeps_the_causal_language_visible():
    html = ui.guardrail(["Association, not causation."])
    assert "ed-guardrail" in html
    assert "Association, not causation." in html
    assert not _has_emoji(html)


def test_process_strip_numbers_stages_in_order():
    html = ui.process_strip([("Collect", "a"), ("Engineer", "b"),
                             ("Train", "c"), ("Predict", "d"), ("Explore", "e")])
    assert html.index("01") < html.index("02") < html.index("05")
    assert html.count("ed-process-step") == 5


def test_numbered_notes_render_two_digit_numbers():
    html = ui.numbered_notes([("Parity", "body"), ("Scope", "body")])
    assert "01" in html and "02" in html
    assert html.count("ed-note-title") == 2


# ----------------------------------------------------------------------------
# Research tables
# ----------------------------------------------------------------------------

def _frame():
    return pd.DataFrame([
        {"Indicator": "Inflation", "Model response (pp)": -0.42},
        {"Indicator": "Population growth", "Model response (pp)": 0.13},
    ])


def test_research_table_right_aligns_numeric_columns():
    html = ui.research_table(_frame())
    assert '<th class="num">Model response (pp)</th>' in html
    assert '<th class="">Indicator</th>' in html


def test_research_table_applies_formats_and_semantics():
    html = ui.research_table(
        _frame(),
        formats={"Model response (pp)": "{:+.2f}"},
        semantic=["Model response (pp)"],
    )
    assert "-0.42" in html and "+0.13" in html
    assert "negative" in html and "positive" in html


def test_research_table_renders_missing_values_without_crashing():
    frame = pd.DataFrame([{"Country": "Chad", "Inflation": float("nan")}])
    html = ui.research_table(frame, formats={"Inflation": "{:.1f}"})
    assert "&ndash;" in html
    assert "nan" not in html.lower()


def test_research_table_can_highlight_and_scroll():
    html = ui.research_table(_frame(), highlight_rows=[1], scroll=True)
    assert 'class="ed-table-wrap scroll"' in html
    assert '<tr class="highlight">' in html


def test_research_table_passes_through_raw_html_columns():
    frame = _frame()
    frame["Response"] = [ui.bar_cell(-0.42, 0.42), ui.bar_cell(0.13, 0.42)]
    html = ui.research_table(frame, raw_html=["Response"])
    assert "<span style=" in html
    assert "&lt;span" not in html


def test_bar_cell_encodes_direction_with_palette_colors():
    positive = ui.bar_cell(0.5, 1.0)
    negative = ui.bar_cell(-0.5, 1.0)
    assert COLORS["terracotta"] in positive
    assert COLORS["plum"] in negative


def test_bar_cell_handles_degenerate_scale():
    assert "width:0.0%" in ui.bar_cell(0.0, 0.0)


# ----------------------------------------------------------------------------
# Africa graphics
# ----------------------------------------------------------------------------

def test_africa_dot_svg_places_one_dot_per_country():
    svg = ui.africa_dot_svg([(10.0, 5.0, 0.9), (30.0, -20.0, 0.2)])
    assert svg.startswith("<svg")
    assert svg.count("<circle") == 2
    assert "rgb(" not in svg


def test_africa_dot_svg_marks_the_highlighted_country():
    svg = ui.africa_dot_svg([(10.0, 5.0, 0.9)], highlight=(10.0, 5.0))
    assert COLORS["terracotta_deep"] in svg
    assert svg.count("<circle") == 2


def test_africa_dot_svg_is_empty_without_points():
    assert ui.africa_dot_svg([]) == ""


def test_africa_map_svg_draws_country_paths():
    countries = [
        {"iso3": "KEN", "rings": [[[34.0, 5.0], [42.0, 5.0], [42.0, -5.0],
                                   [34.0, -5.0]]]},
        {"iso3": "TZA", "rings": [[[30.0, -1.0], [40.0, -1.0], [40.0, -12.0],
                                   [30.0, -12.0]]]},
    ]
    svg = ui.africa_map_svg(countries, {"KEN": 0.9, "TZA": 0.2}, highlight="KEN")
    assert svg.count("<path") == 2
    assert COLORS["ink"] in svg          # highlight stroke
    assert "Z" in svg                    # closed rings


def test_africa_map_svg_uses_inactive_tone_without_values():
    countries = [{"iso3": "KEN", "rings": [[[0, 0], [1, 0], [1, 1], [0, 1]]]}]
    svg = ui.africa_map_svg(countries, {})
    assert COLORS["inactive"] in svg


def test_africa_map_svg_is_empty_without_geometry():
    assert ui.africa_map_svg([]) == ""
