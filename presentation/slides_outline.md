# Africa Growth Explorer - Presentation Outline

**5-10 Minute Presentation / Demo Script**

> Numbers on slides 3, 5 and 6 are pasted from `reports/generated/` (built by
> `scripts/build_report_assets.py` from committed model artifacts). Do not
> retype them by hand.

---

## Slide 1: Title Slide (30 seconds)

**Africa Growth Explorer**
*A Machine Learning Decision-Support System Using World Bank Development Indicators*

**Core Question:** Can recent development indicators predict near-term GDP per capita growth across African countries?

**One-line answer established in this project:** *not significantly beyond the unconditional mean — and here is a protocol rigorous enough to prove it.*

**Presenter:** [Your Name] · **Date:** [Date] · **Program:** FlyRank ML Engineering Internship Capstone

---

## Slide 2: Problem & Motivation (60 seconds)

### Why This Matters
- 54 African UN member states, diverse growth trajectories
- Development analysts need screening tools, not just predictions
- A prediction is only useful if you can say how much better it is than "predict the average" — most portfolios skip that step

### Decision-Support Framing
- **Not:** causal policy engine
- **Yes:** statistical-association explorer + a *baseline-gated* forecasting pipeline
- **User:** analysts, researchers, policy teams, NGOs

---

## Slide 3: Data & Target (60 seconds)

### Source
World Bank WDI — 14 candidate indicators (13 survive coverage filtering + current-year growth column), 52 countries in the committed panel (config lists 54 + ESH; MUS/SDN arrive with next re-ingestion), 2000-2024

### Target Construction
```
Features at year t  →  Target at year t+1
Ghana 2019 indicators  →  Ghana 2020 GDP growth
```

### Temporal Split — stated in FEATURE years, with TARGET years made explicit
| Split | Feature years | Target years | n |
|---|---|---|---|
| Train | 2000–2017 | 2001–2018 | 905 |
| Validation | 2018–2020 | **2019–2021** (includes COVID) | 150 |
| Test | 2021–2023 | **2022–2024** (sealed; read once) | 150 |

---

## Slide 4: Methodology — the protocol IS the product (60 seconds)

### Selection Pipeline (`scripts/finalize_model.py`, pre-registered order)
```
Raw WDI CSV → explicit ISO3 Africa filter (no MENA substring trap)
→ country-year panel + duplicate policy (spec §7)
→ next-year target via grouped shift
→ coverage filter (≥60% on TRAINING rows only)
→ expanding-window CV INSIDE the train period (spec §9):
     HGB depth {2,3} × lr {0.01,0.03,0.05} × iter {100,200}, early_stopping=True
     Ridge α {1…3000}
→ CV-best of each family scored on VALIDATION
→ BASELINE GATE on validation (fail ⇒ no artifacts, exit 2)
→ pre-registered refit policy: train-only (COVID regime-mismatch rationale)
→ TEST touched exactly once: metrics + paired bootstrap significance
→ provenance-stamped artifacts (sha256, git commit, versions)
```

### Models
1. **Ridge** — linear benchmark, direction-carrying standardized coefficients
2. **HistGradientBoosting** — deployed (won validation, passed gate)

### Baselines (all three from spec §10)
- Global mean (training average)
- Persistence (current year = next year)
- Country historical mean (expanding, no future data)

---

## Slide 5: Results — the honest scoreboard (90 seconds)

### Test Set Metrics (target years 2022-2024, 150 observations)

| Model | Split | MAE | RMSE | R² | Dir. acc | Majority rate | Dir. skill |
|---|---|---:|---:|---:|---:|---:|---:|
| Global mean baseline | test | 1.90 | 2.84 | -0.00 | 80.7% | 80.7% | 0.0 pp |
| Persistence baseline | test | 2.23 | 4.52 | -1.54 | 77.3% | 80.7% | -3.3 pp |
| Country historical mean | test | 1.94 | 2.88 | -0.03 | 78.0% | 80.7% | -2.7 pp |
| **HGB (deployed)** | test | **1.82** | **2.79** | **0.03** | 80.7% | 80.7% | 0.0 pp |

(Persistence R²/MAE on its fair-comparison subset; see report §7.)

### The number that decides the story
> **Paired bootstrap vs global mean: +0.07 pp, 95% CI [−0.04, +0.19] — spans zero.**
> Model = statistical parity with the unconditional mean. This is the finding.

### How to read "directional accuracy" (H4)
80.7% of test years have non-negative growth, so any always-positive constant scores 80.7% by construction. Accuracy **without** the majority rate is class prior, not skill; we report skill and balanced accuracy (51.3%) alongside.

---

## Slide 6: Interpretation — what the model actually learned (60 seconds)

### Permutation importance on validation, with 95% CIs (spec §13)
- **2 of 14 features distinguishable from zero:** GDP per capita (0.046 [0.018, 0.085]) and population growth (0.017 [0.000, 0.037])
- The other 12 — including electricity and inflation — have intervals **crossing zero**: attribution is noise at this signal level
- Permutation importance is magnitude-only: **no "positive/negative driver" column** (earlier drafts had one; retracted as a category error)

### Direction from Ridge standardized coefficients (CV-best α=3000)
- All |coefficients| ≤ 0.14 — even the linear association structure is faint
- α pinned at the grid's largest value: shrinkage toward the mean is the best the linear model can do
- Association only, never causal (confounding, reverse causality, omitted shocks)

---

## Slide 7: Live Demo — Streamlit App (3 minutes)

### Page 1: Project Overview
- Problem, data, model card, headline metrics **with** significance caption, causal disclaimer

### Page 2: Explore Africa
- Country selector → growth trend (observed vs next-year target), indicator trends, regional comparison

### Page 3: Model Performance
- Significance banner; test + validation baseline tables with majority-rate columns
- Actual-vs-predicted, residuals, CI-gated importance chart, noise table, Ridge-direction panel, by-year metrics

### Page 4: Scenario Explorer ← **Core Decision-Support Feature**
1. Select country + reference year → baseline feature values (with imputation notice)
2. Sliders bounded by **training-window** range; warning band = training P1–P99 (inflation warns above ~49.5, not 92)
3. Out-of-band defaults are clamped **and disclosed**
4. Prediction delta table = one-at-a-time model re-runs ("Individual effect (pp)"), with explicit non-additivity caption
5. Causal disclaimer prominent

*Demo narration uses only live app output; no fixed numbers are scripted here — a previous draft's script figures were illustrative and have been removed.*

---

## Slide 8: Technical Rigor (60 seconds)

### What a serious ML review should check, and what this repo does
- **Leakage-free protocol:** coverage filter on train rows; in-pipeline imputation/log transform; target via grouped shift; test sealed until a single scoring pass
- **Baseline gate:** a worse-than-constant model cannot ship (enforced pre-artifact, exits non-zero)
- **Real expanding-window CV** with committed result tables (`models/cv_results_*.csv`)
- **Pre-registered refit policy** (train-only; COVID regime mismatch) with train+val reported as sensitivity — decided before test was read
- **Significance testing:** paired bootstrap CI on the headline comparison, reported whatever it says
- **Provenance:** panel SHA-256, git commit, library versions, split target-years in metadata; bit-deterministic re-run
- **Test suite:** 60+ tests including adversarial regression guards (corrupt the test split ⇒ selection byte-identical; gate-fail ⇒ no artifacts)
- **Notebooks executed** (no hardcoded paths), consistent with deployment by construction (they *load* the deployment)

### Engineering
- Artifact-driven app: cached model/data, no retraining, no network calls
- Packaging fixed: `pip install -e .` exposes `src.*`; runtime and dev requirements separated

---

## Slide 9: What This Taught Us — null result as deliverable (60 seconds)

### The finding
- At annual frequency, pooled WDI aggregates carry **no statistically significant** information about next-year growth beyond the mean. The strongest, honestly selected model reaches test parity (1.82 vs 1.90, CI spans zero).
- This is *substantive*: annual country-level WDI levels/flows are too slow-moving to resolve short-run growth; variance is dominated by shocks (every worst error is a conflict/oil/tourism event year).

### Why believe *this* null and not a previous non-null?
- The same dataset previously "produced" a deployed model worse than a constant, a test-set-selected winner, and a report with fabricated statistics — all removed by fixing the protocol
- Removing the leaks removed the mirage: the negative result is what honest machinery reports

### When to trust / not trust the tool
| Trust for | Don't trust for |
|---|---|
| Cross-country descriptive comparison | Budget allocation from point predictions |
| Understanding what annual WDI can/cannot support | Causal policy claims from scenario deltas |
| Template for gated, pre-registered modeling projects | Anything requiring sub-95%-CI discrimination of 0.07pp effects |

---

## Slide 10: Future Work (30 seconds)

1. **Higher-frequency data** — nightlights, mobile money, trade flows (the parity result argues for *better signal*, not *more models*)
2. **Event-aware features** — commodity prices, political instability, climate shocks
3. **Prediction intervals** in-app (bootstrap machinery is already in `src/evaluate.py`)
4. **Re-ingest WDI** for full 54-country coverage, re-run the protocol end-to-end
5. **Hierarchical / panel benchmark** — our CV essentially rediscovers shrink-toward-mean; make it explicit

---

## Slide 11: Key Takeaways (30 seconds)

1. **A defensible null beats an undefended win** — the protocol is the contribution
2. **Gate your models** — "does it beat the baseline it claims to replace?" belongs in the build, not the discussion section
3. **Generate every document number from artifacts** — the report and the metadata cannot drift by construction
4. **Decision-support honesty sells** — guardrails, majority-rate-aware metrics, and CI-gated attribution make the tool usable *because* it doesn't oversell
5. **Reproducibility** — pinned panel hash, fixed seeds, bit-deterministic finalize, executed notebooks

---

## Slide 12: Links & Contact (15 seconds)

- **GitHub:** this repository (code, artifacts, tests, executed notebooks, report + PDF)
- **App:** verified locally via `streamlit run app.py` (see README Deployment; a public Streamlit Cloud URL is not part of this submission)
- **Report:** `reports/capstone_report.md` / `reports/capstone_report.pdf`
- **Contact:** [your email]

---

## Talking Points for Q&A

- **Why HGB over Ridge?** CV-best HGB beat CV-best Ridge on validation MAE (3.89 vs 4.00) and cleared the gate; both beat the validation baselines narrowly.
- **Is "parity" a failure?** No — it is a quantified, statistically tested statement about predictability, obtained by a protocol that had no way to fake it.
- **Why early stopping mattered?** The previous cycle's overfit came from sklearn `early_stopping="auto"` being inert at n=905; explicit `True` bounds the model (deployed runs 45 of 200 iterations).
- **Why pre-register the refit?** Validation targets include the COVID crash; refitting across it biases test-period predictions (sensitivity shows MAE 2.03 vs 1.82 and −0.86pp bias).
- **Why not tune harder?** Test-set fishing is how the last cycle shipped a bad model; the grid is compact, pre-registered, and scored only inside training years.
- **How often retrain?** Annually with WDI updates; gate + significance must pass each time.
- **Can this inform policy?** Only as screening context; causal claims need experimental/quasi-experimental designs (spec §2 boundary).

---

## Appendix Slides (if time permits)

- A1: Expanding-window CV tables (`models/cv_results_*.csv`)
- A2: Bootstrap CIs + paired significance (report §7)
- A3: Worst-error rows (all real; report §7)
- A4: Feature-selection audit (coverage filter, `SE.SEC.ENRR`)
