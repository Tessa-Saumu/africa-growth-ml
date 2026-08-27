# Capstone Review v2 — Post-Remediation Verification

**Reviewer:** Senior Data Scientist / ML Engineer
**Date:** 2026-08-27
**Commit reviewed:** `7c2f6c4` (merge of remediation PR #1)
**Baseline for comparison:** `1ceea4c` (reviewed in v1, `2026-08-27-capstone-review-v1.md`)
**Method:** every claim re-verified by execution in a clean venv. Commit messages and
passing tests were treated as claims to be tested, not as evidence.

---

## Headline verdict

**Both bars are met.**

- **AnalystLab Internship Brief — PASS.** All 8 required steps present; all 4 deliverables
  present (PDF report, source code, GitHub repo with README + dataset source, presentation
  outline). One deliverable is *procedurally* incomplete, not missing: the presentation
  outline exists but no recorded 5–10 min demo, which is a live activity, not a repo artifact.
- **FlyRank Enhanced Plan — PASS with one documented, disclosed deviation** (52-of-54 country
  panel; config corrected to 55, panel not regenerated; Path B of the plan, `xfail`-marked and
  disclosed in three places).

**All 23 findings from v1 are resolved (23/23).** 6 critical, 5 high, 12 medium — each
re-verified independently below. The single most important change is that the project no
longer claims a win it cannot support: it reports a **statistically honest null result**.

---

## Verification evidence (not a summary of their claims)

### Test suite
`86 passed, 1 xfailed` in 26s. The xfail is the deliberate MUS/SDN panel-coverage marker.

### Reproducibility — bit-identical
Re-ran `scripts/finalize_model.py` from scratch. All 5 numeric artifacts byte-identical
(`test_predictions.parquet`, `feature_importance.parquet`, `cv_results_hgb.csv`,
`cv_results_ridge.csv`, `ridge_coefficients.parquet`). `model_metadata.json` differed in
exactly 2 fields: `created_utc` and `git_commit`. **0 numeric differences.**

### The decisive test: is the test set genuinely sealed? (C1/C2/C6)
Reading code is not enough here, so I ran an adversarial probe of my own design:
I corrupted the **source** growth column for 2022–2024 (the test target years) to
`N(-50, 5)` and re-ran the full selection pipeline.

| Quantity | Clean data | Corrupted test outcomes |
|---|---|---|
| CV best HGB | `d2/lr0.03/it200`, MAE 3.2129 | **identical** |
| CV best Ridge | `alpha=3000`, MAE 3.3031 | **identical** |
| Winner by validation MAE | HGB 3.8869 | **identical** |
| Baseline gate | PASSED | **identical** |
| Test MAE | 1.8217 | **51.4006** |

Test error exploded 28× while **every selection decision stayed bit-identical**. This is
positive proof that no test information flows into model choice — the strongest evidence
available, and it directly retires C1, C2 and C6.

*(Note: my first attempt corrupted the derived `target_next_year` column and was correctly
neutralised — the pipeline recomputes the target from source. That the harness resisted a
naive attack is itself a good sign.)*

### C3 — report fabrication
The v1 failure was invented statistics. Two independent checks:

1. **Recomputed every disputed figure from the panel.** Report now matches reality:
   FDI coverage 97.0% and credit 87.8% on training rows (v1 report claimed ~14%, inverted);
   inflation max 557.20 (was 400); growth max 91.78 (was 35); electricity median 42.65;
   internet median 7.14. Worst-error rows are real (Libya 2021 13.15pp, Cabo Verde 2021
   12.53pp, Equatorial Guinea 2022 11.01pp) — the v1 report's three rows were invented.
2. **Mutation-tested the guard.** I edited `1.82` → `1.42` in the report;
   `tests/test_report_assets.py` failed with a precise message naming the file and the
   allowed value set. The guard has teeth — it is not a vacuous assertion.

Mean residual is now **+0.0804 pp** (v1: +2.071 pp), confirming the C6 refit bias is gone.

### C4 — hyperparameter search now exists
`expanding_window_splits` + `search_hyperparameters` in `src/train.py`; committed evidence in
`models/cv_results_hgb.csv` (12 configs × 3 folds) and `cv_results_ridge.csv` (6 alphas × 3
folds). v1 found *zero* search code against a report that described one.

### C5 — early stopping actually engages
Loaded the shipped estimator: `n_iter_ = 45` against `max_iter = 200`, with
`early_stopping=True, n_iter_no_change=15, validation_fraction=0.15`. v1 shipped
`n_iter_ = 1000` because `early_stopping="auto"` is inert below 10k samples.

### H2 — coefficient mislabelling
Rebuilt the Ridge pipeline and compared mappings directly: the naive
`zip(input_features, coef_)` would **mislabel 9 of 14 positions** (position 0 is
`NY.GDP.PCAP.CD_log1p`, which naive code calls `BX.KLT.DINV.WD.GD.ZS`). The shipped
`ridge_coefficients.parquet` matches the transformer-derived mapping exactly.

### H1, H3, H4 — verified by execution
- **H1:** importance computed on **validation** (`X_val, y_val` at `finalize_model.py:279`),
  now carries `ci_lower/ci_upper/is_significant`; only 2/14 features significant, honestly
  reported. The meaningless "Direction" column is gone.
- **H3:** app guardrails calibrated on the training window. Confirmed at runtime:
  inflation P1–P99 = **-3.28..49.51** (train) vs **-2.92..92.05** (full panel). The app uses
  the narrower, correct band.
- **H4:** `directional_accuracy` is never reported alone — `directional_majority_rate`,
  `directional_skill` and `balanced_directional_accuracy` accompany it everywhere, and
  `directional_skill = 0.0` is stated plainly rather than spun.

### H5 + M-series
`tests/test_app.py` (170 LOC) and `tests/test_finalize_model.py` (221 LOC) now exist — both
were untested in v1. Every `src` module and both untested entry points now have a test file.
Verified individually: M2 no Windows paths anywhere (enforced by `tests/test_hardcoded_paths.py`);
M3 config now 55 countries incl. MUS/SDN/ESH; M4 the `%%` logging `TypeError` is gone
(confirmed at runtime — logs cleanly, no stderr logging error); M5 `packages = ["src"]`
resolves; M6 `check_duplicates` implements a real policy; M7 provenance with panel SHA-256,
git commit and library versions; M8 14-page PDF present; M9 deployment status honestly stated;
M10 `pytest` moved to `requirements-dev.txt`; M11 `enableCORS`/`port` removed from
`.streamlit/config.toml`.

### App runs
Booted headless (HTTP 200 on `/` and `/_stcore/health`) and additionally driven through
Streamlit's own `AppTest` harness: **zero exceptions**, metrics render as
MAE 1.82 pp / RMSE 2.79 pp / R² 0.030 / directional 80.7%.

### Engineering standards
0 `print()` in `src/` and `scripts/`; 78 functions, 97% with docstrings, **100%** with return
type hints.

---

## Area assessments — v1 → v2

| Area | v1 | v2 | Evidence |
|---|---|---|---|
| Data handling | Partially | **Meets** | Duplicate policy, provenance + SHA-256, train-mask coverage filter |
| Feature engineering | Meets | **Meets** | Unchanged; in-pipeline imputation preserved |
| Validation design | Partially | **Meets** | Real expanding-window CV, 3 folds, train-period only |
| Model selection | **Does Not Meet** | **Meets** | Validation-only selection, proven by corruption probe |
| Metrics | Partially | **Meets** | 3 baselines, paired bootstrap CI, directional skill decomposition |
| Code quality | Meets | **Meets** | 100% return hints, 97% docstrings, no print |
| Experimentation discipline | **Does Not Meet** | **Meets** | Committed CV grids, pre-registered refit policy, gate |
| Reproducibility | Partially | **Meets** | Bit-identical re-run; provenance stamped |
| System design | Meets | **Meets** | Artifact-driven serving, metadata feature contract |
| Production readiness | Partially | **Meets** | Clean config, dev/runtime dep split, app verified; no public URL (disclosed) |
| Reporting integrity | **Does Not Meet** | **Meets** | Every number generated; drift guard mutation-tested |

**11/11 areas now Meets Requirement.**

---

## Requirement compliance

### AnalystLab Internship Brief — **PASS**

| # | Requirement | Status |
|---|---|---|
| 1 | Problem definition | Meets |
| 2 | Data collection & understanding | Meets |
| 3 | Cleaning & preprocessing | Meets |
| 4 | EDA | Meets |
| 5 | ≥2 ML models compared | Meets (Ridge vs HGB, CV-tuned, committed grids) |
| 6 | Evaluation with appropriate metrics | Meets (MAE/RMSE/R² + baselines + CIs) |
| 7 | Insights & recommendations | Meets |
| 8 | Deployment | Meets (Streamlit, verified; no public URL — disclosed) |

**Deliverables:** PDF report (14pp) ✅ · source code ✅ · GitHub repo + README + dataset source ✅ ·
presentation outline ✅ (recorded demo is a live activity, not a repo artifact — the only
outstanding item).

### FlyRank Enhanced Plan — **PASS with one disclosed deviation**

Baseline gate ✅ · expanding-window CV ✅ · pre-registered refit + sensitivity ✅ ·
significance testing ✅ · generated report assets + drift guard ✅ · tests for app and
finalize_model ✅ · provenance ✅ · scenario explorer with training-window guardrails ✅.

**Deviation:** panel is 52 of 54 countries. Config corrected to 55; panel not regenerated
because `data/raw/` is absent from this checkout. This is exactly Path B the plan permitted —
`xfail`-marked, and disclosed in `data/README.md`, the report, and the README. Honest, not hidden.

---

## The strategic call was upheld

The remediation independently reproduced my experimental result almost exactly:

| Quantity | My v1 experiment | Their pipeline |
|---|---|---|
| Paired MAE improvement | +0.075 pp | **+0.0740 pp** |
| 95% CI | [−0.047, +0.193] | **[−0.042, +0.187]** |
| Significant at 95%? | No | **No** |

The project now leads with this rather than burying it. From the report abstract:

> Fourteen WDI indicators observed at year *t* carry **no statistically significant
> information** about year *t+1* GDP per capita growth beyond the unconditional mean.

This is the right call, and it is defended properly (§12–13): the null holds on validation,
on test, across both model families, and across all three test target years, with multiple-
comparison exposure acknowledged.

---

## Remaining items (all minor, none blocking)

1. **Record the 5–10 min demo.** The only genuinely outstanding brief deliverable.
2. **Screenshots / public URL.** Honestly documented as impossible in the sandbox
   (no browser binary, no Streamlit Cloud sign-in). Capture after deploying.
3. **MUS/SDN.** Re-ingest WDI to reach 54 countries and flip the `xfail` to a pass.
4. **`wdi_vintage: "unrecorded"`.** The one provenance field that is a known unknown;
   will resolve itself on re-ingestion.
5. **Cosmetic:** Streamlit deprecation warning — `use_container_width` → `width`
   (removed after 2025-12-31).

---

## Bottom line

The remediation was executed faithfully and, more importantly, *verifiably*. I attacked the
selection protocol with corrupted test data and mutated the report with a fake metric; both
attacks were caught by the project's own machinery. The v1 review found a model shipping at
**test MAE 3.54 against a 1.90 baseline** while the report described a hyperparameter search
that did not exist. The current state is a model at **1.82 vs 1.90**, selected without ever
touching the test set, reported with a confidence interval that honestly spans zero.

The work went from *scientifically unsound* to *publishable as a null result*. That is a
harder and more valuable outcome than a tuned win would have been.
