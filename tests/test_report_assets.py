"""Guards against C3: hand-written numbers drifting from artifacts.

Every MAE figure quoted in a document must equal a value present in the
generated `reports/generated/metrics.json` (which itself is derived only from
committed model artifacts). Worst-error country names in the report must be
real rows from the frozen test predictions.
"""
import json
import re
from pathlib import Path

import pytest

DOCS = ["README.md", "reports/capstone_report.md", "presentation/slides_outline.md"]


@pytest.fixture(scope="module")
def metrics():
    p = Path("reports/generated/metrics.json")
    if not p.exists():
        pytest.skip("Run scripts/build_report_assets.py first")
    return json.loads(p.read_text())


def _all_mae_values(obj, out):
    """Recursively collect every float stored under a key containing 'mae'."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (int, float)) and "mae" in k.lower():
                out.add(round(float(v), 2))
            elif isinstance(v, list) and "mae" in k.lower():
                out.update(round(float(x), 2) for x in v if isinstance(x, (int, float)))
            else:
                _all_mae_values(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _all_mae_values(v, out)


def test_documents_quote_correct_test_mae(metrics):
    """Any 'MAE <number>' in a document must match an artifact-derived value.

    DEVIATION (flagged): the remediation plan's draft allowed exactly three
    values (deployed/global-mean/persistence test MAE). Documents also
    legitimately quote *validation* and CV fold MAEs — all of which live in
    metrics.json — so the allow-list is every MAE-valued number the generator
    emitted from artifacts. Fabricated values (the old 'MAE 3.54 vs 1.90'
    framing, invented grid MAEs, etc.) still fail.
    """
    allowed = set()
    _all_mae_values(metrics, allowed)
    assert allowed, "metrics.json contains no MAE values"

    pattern = re.compile(r"MAE[^0-9\-]{0,20}(\d+\.\d{2})")
    for doc in DOCS:
        text = Path(doc).read_text(encoding="utf-8")
        for found in pattern.findall(text):
            assert float(found) in allowed, (
                f"{doc} quotes MAE {found}; artifacts allow {sorted(allowed)}")


def test_no_fabricated_worst_error_countries(metrics):
    """C3: report claimed Libya +35pp etc. Worst-error names must be real."""
    real = {r["country_name"] for r in metrics["worst_errors"][:10]}
    report = Path("reports/capstone_report.md").read_text(encoding="utf-8")
    m = re.search(r"###\s+Worst\s+errors.*?(?=\n##)", report, re.S | re.I)
    assert m, ("No worst-errors section found in the report. This guard must "
               "not silently skip: if the section is renamed, update the "
               "pattern rather than losing the check.")
    section = m.group(0)

    # Prose form: '**Libya 2021**: actual ...'
    claimed = set(re.findall(r"\*\*?([A-Z][a-zA-Z ]+?)\s+20\d{2}", section))
    # Table form: '| Libya | 2021 | ...'
    claimed |= {c.strip() for c in
                re.findall(r"^\|\s*([A-Z][a-zA-Z ]+?)\s*\|\s*20\d{2}\s*\|",
                           section, re.M)}
    assert claimed, "Worst-errors section lists no country-year rows"
    for name in claimed:
        assert name.strip() in real, f"'{name}' is not in the real top-10"


def test_report_never_restores_overturned_claims():
    """The specific fabricated claims from the review must be gone."""
    banned = [
        ("reports/capstone_report.md", "-60% to +35%"),
        ("reports/capstone_report.md", "~55%"),
        ("reports/capstone_report.md", "median ~15%"),
        ("reports/capstone_report.md", "400%"),
        ("reports/capstone_report.md", "Worst coverage: FDI"),
        ("reports/capstone_report.md", "max_iter ∈ {500, 1000}"),
        ("reports/capstone_report.md", "Mean residual ≈ 0 (unbiased)"),
        ("README.md", "strongest positive predictor"),
        ("README.md", "python -m src.train"),
        ("presentation/slides_outline.md", "better than random"),
    ]
    for doc, needle in banned:
        text = Path(doc).read_text(encoding="utf-8")
        assert needle not in text, f"{doc} still contains removed claim: {needle!r}"


def test_directional_accuracy_never_alone(metrics):
    """H4: wherever directional accuracy is quoted for the deployed model in the
    README/report, the majority-class rate must appear in the same document."""
    docs = [Path("README.md"), Path("reports/capstone_report.md")]
    for p in docs:
        text = p.read_text(encoding="utf-8")
        if re.search(r"[Dd]irectional", text):
            assert re.search(r"majority", text), f"{p} quotes directional accuracy without majority rate"


def test_metrics_json_traceable_to_model_metadata(metrics):
    """Generator must not invent: headline numbers equal models/model_metadata.json."""
    meta = json.loads(Path("models/model_metadata.json").read_text(encoding="utf-8"))
    assert metrics["winner_test"]["mae"] == pytest.approx(
        meta["metrics"]["winner_test"]["mae"], abs=1e-12)
    assert metrics["significance"] == meta["significance"]
    assert metrics["gate"] == meta["gate"]


def test_generated_tables_exist_and_nonempty(metrics):
    """Every promised asset is written by the generator."""
    gen = Path("reports/generated")
    for name in ["table_model_comparison.md", "table_feature_importance.md",
                 "table_ridge_coefficients.md", "table_yearly_metrics.md",
                 "table_worst_errors.md", "table_eda_summary.md",
                 "table_correlations.md", "table_cv_results.md"]:
        p = gen / name
        assert p.exists() and len(p.read_text(encoding="utf-8").splitlines()) > 2, name
    for fig in ["actual_vs_predicted.png", "residuals.png",
                "feature_importance.png", "correlation_heatmap.png",
                "eda_missingness_heatmap.png", "eda_feature_distributions.png",
                "eda_correlation_matrix.png", "modeling_actual_vs_predicted.png",
                "modeling_residuals.png", "modeling_feature_importance.png"]:
        assert (Path("figures") / fig).exists(), fig
