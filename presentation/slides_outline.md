# Africa Growth Explorer — Presentation Outline

**Format:** 5–10 minute presentation / demo video
**Scripted runtime: 9 minutes 15 seconds**, leaving 45s of buffer inside the 10-minute ceiling.

> Every number on slides 3, 5 and 6 is pasted from `reports/generated/`, which
> `scripts/build_report_assets.py` builds from committed model artifacts.
> Do not retype them by hand. `tests/test_report_assets.py` fails the build if
> any document quotes a metric the artifacts do not support.

---

## Required-section coverage

The brief names six items. Each maps to a numbered slide:

| Required item | Slide | Time |
|---|---|---:|
| Problem Statement | 2 | 45s |
| Dataset Used | 3 | 50s |
| Analysis Process | 4 | 70s |
| Model Performance | 5 | 85s |
| Key Insights | 6 | 65s |
| Recommendations | 8 | 55s |
| *(supporting: title, demo, rigor, close)* | 1, 7, 9, 10 | 175s |

---

## Slide 1: Title (20 seconds)

**Africa Growth Explorer**
*Predicting next-year GDP per capita growth across African economies from World Bank development indicators*

**Question:** can recent development indicators predict near-term growth?

**Answer, established under a pre-registered protocol:** no — not significantly beyond the unconditional mean. The contribution is a pipeline rigorous enough to establish that rather than hide it.

**Presenter:** [Your Name] · **Date:** [Date] · **Program:** AnalystLab Data Science Internship Capstone

> **Say:** "I'll give you the conclusion first, because it shapes everything after it. The model reaches parity with predicting the average, and I can show you why that's the correct answer rather than a failed one."

---

## Slide 2: Problem Statement (45 seconds)

### The task
Predict GDP per capita growth for year *t+1* from development indicators observed at year *t*, across 54 African UN member states.

### Why it's worth doing carefully
Development analysts need screening tools. But a growth prediction is only useful if you can state how much better it is than "predict the average" — and that comparison is the step most portfolio projects skip.

### Framing
- **Not:** a causal policy engine
- **Is:** a statistical-association explorer plus a baseline-gated forecasting pipeline
- **Users:** analysts, researchers, policy teams, NGOs

### Success criterion, fixed before modelling
Beat the global-mean, persistence, and country-historical-mean baselines **on validation**, with the margin tested for significance on a sealed test set. A model that cannot clear that bar does not ship.

---

## Slide 3: Dataset Used (50 seconds)

### Source
World Bank **World Development Indicators** (WDI), 2000–2024. Public, no licence restriction, cited in `data/README.md`.

### Shape
- 14 candidate indicators across 6 themes — 13 survive coverage filtering, plus the current-year growth column
- **52 countries** in the committed panel; config lists all 54 UN member states plus ESH. Mauritius and Sudan are absent from this panel vintage and arrive on the next re-ingestion. Stated here rather than rounded to "54"
- 1,300 country-year rows; 1,205 usable after target construction

### Target construction
```
Features at year t  →  Target at year t+1
Ghana 2019 indicators  →  Ghana 2020 GDP per capita growth
```

### Temporal split — feature years, with target years made explicit
| Split | Feature years | Target years | n |
|---|---|---|---|
| Train | 2000–2017 | 2001–2018 | 905 |
| Validation | 2018–2020 | **2019–2021** (contains COVID) | 150 |
| Test | 2021–2023 | **2022–2024** (sealed, read once) | 150 |

> **Say:** "The target years matter more than the feature years. Validation contains the COVID crash; test is the recovery. That mismatch drives a design decision on the next slide."

---

## Slide 4: Analysis Process (70 seconds)

### Pipeline — `scripts/finalize_model.py`, executed in this fixed order
```
Raw WDI CSV
→ explicit ISO3 Africa filter (no MENA substring trap)
→ country-year panel + duplicate policy
→ next-year target via grouped shift
→ coverage filter ≥60%, computed on TRAINING rows only
→ expanding-window CV inside the train period:
     HGB  depth {2,3} × lr {0.01,0.03,0.05} × iter {100,200}, early_stopping=True
     Ridge α {1 … 3000}
→ CV-best of each family scored on VALIDATION
→ BASELINE GATE on validation — fail ⇒ no artifacts written, exit 2
→ pre-registered refit: train-only
→ TEST read exactly once: metrics + paired bootstrap significance
→ provenance-stamped artifacts (SHA-256, git commit, library versions)
```

### EDA findings that shaped the model
- Development levels — electricity, internet, urbanisation, life expectancy, GDP per capita — correlate at **0.45–0.71**. Five measurements of roughly one latent variable, not five signals
- Current-year growth is uncorrelated with every level indicator (|r| ≤ 0.13)
- Missingness is structural, clustering in specific country blocks, so the coverage filter runs on training rows only

### Models compared
1. **Ridge** — linear benchmark, standardised coefficients carry direction
2. **HistGradientBoosting** — deployed; won validation, cleared the gate

### Three baselines
Global mean · persistence · country historical mean (expanding, no future data)

> **Say:** "The refit policy was pre-registered. Validation spans COVID, test is the recovery — refitting across that regime break biases predictions downward. I fixed train-only before reading test, and I report the alternative as sensitivity."

---

## Slide 5: Model Performance (85 seconds)

### Test set — target years 2022–2024, n=150

| Model | MAE | RMSE | R² | Dir. acc | Majority rate | Dir. skill |
|---|---:|---:|---:|---:|---:|---:|
| Global mean baseline | 1.90 | 2.84 | −0.00 | 80.7% | 80.7% | 0.0 pp |
| Persistence baseline | 2.23 | 4.52 | −1.54 | 77.3% | 80.7% | −3.3 pp |
| Country historical mean | 1.94 | 2.88 | −0.03 | 78.0% | 80.7% | −2.7 pp |
| **HGB (deployed)** | **1.82** | **2.79** | **0.03** | 80.7% | 80.7% | 0.0 pp |

*(Persistence scored on its fair-comparison subset — report §7.)*

### The number that decides the story
> **Paired bootstrap vs global mean: +0.07 pp, 95% CI [−0.04, +0.19].**
> The interval contains zero. The model is at statistical parity with predicting the average. That is the finding.

### Validation, where selection actually happened
Ridge 4.00 · HGB **3.89** · global-mean baseline 4.04 — gate passed by 3.7%, which is inside noise at n=150. The null shows up here too.

### Reading directional accuracy honestly
80.7% of test targets are non-negative, so any always-positive constant scores 80.7% by construction. Accuracy without the majority rate is a class prior, not skill. Reported alongside: **skill 0.0 pp, balanced accuracy 51.3%.**

> **Say:** "R² is +0.03. That is not a model that explains growth — it's a model that has correctly learned to predict the mean and stop."

---

## Slide 6: Key Insights (65 seconds)

### 1. Only 2 of 14 features are distinguishable from noise
Permutation importance on validation, with 95% CIs:
- GDP per capita **0.046** [0.018, 0.085]
- Population growth **0.017** [0.000, 0.037]
- The other twelve — including electricity and inflation — have intervals crossing zero

Scale check: the larger effect is 0.046 MAE-degradation units against a validation MAE near 3.9. About one percent. Detectable, negligible.

### 2. Importance is magnitude-only
No positive/negative driver column. Permutation importance measures degradation from scrambling and carries no sign. An earlier draft had that column; it was a category error and was removed.

### 3. Even the linear structure is faint
Ridge α settled at 3000 — the largest value in the grid. Maximum shrinkage toward the mean is the best a linear model can do here. All |coefficients| ≤ 0.14.

### 4. The errors are events, not model failure
Every worst-error case is a shock year: Libya 2021 (13.15 pp), Cabo Verde 2021 (12.53 pp), Equatorial Guinea 2022 (11.01 pp). Conflict, tourism collapse, oil. No annual indicator panel anticipates these.

### 5. Why believe this null
The same dataset previously produced a "working" model that was in fact worse than a constant — its test error was nearly double the global-mean baseline's — with the winner selected on the test set and fabricated statistics in the report. Fixing the protocol removed the mirage. The null is what the honest machinery reports.

---

## Slide 7: Live Demo — Streamlit App (110 seconds)

**Page 1 — Overview:** problem, data, model card, headline metrics with the significance caption, causal disclaimer.

**Page 2 — Explore Africa:** country selector, growth trend (observed vs next-year target), indicator trends, regional comparison.

**Page 3 — Model Performance:** significance banner; test and validation baseline tables with majority-rate columns; actual-vs-predicted; residuals; CI-gated importance with the noise table; Ridge direction panel; per-year metrics.

**Page 4 — Scenario Explorer** ← the decision-support feature
1. Country + reference year → baseline feature values, with an imputation notice
2. Sliders bounded by the **training window**; warning band is training P1–P99, so inflation warns above ~49.5 rather than 92
3. Out-of-band defaults are clamped **and the clamp is disclosed**
4. Delta table = one-at-a-time model re-runs, captioned as non-additive
5. Causal disclaimer stays on screen

*Narrate from live output only. No figures are scripted here — an earlier draft's illustrative numbers were removed.*

---

## Slide 8: Recommendations (55 seconds)

### For anyone using this tool
1. **Do not allocate funding from these point predictions.** They are statistically indistinguishable from a constant. This is the first recommendation because it is the one with consequences.
2. **Use it for descriptive comparison** of development profiles — the Explore page. That is what the data supports.
3. **Treat every scenario delta as a model response**, never an intervention effect.

### For the next modelling cycle
4. **Prioritise higher-frequency signal** — nightlights, port and air-traffic data, mobile-money flows, survey expectations — over more annual WDI indicators. Parity argues for better signal, not more models.
5. **Keep the baseline gate and the pre-registered protocol.** That is the reusable asset from this project.
6. **Re-ingest WDI** to restore Mauritius and Sudan before quoting country coverage externally.
7. **Watch for regime breaks.** This null is partly a statement about a 25-year window containing three very different macro regimes.

---

## Slide 9: Engineering & Rigor (40 seconds)

- **Leakage control:** coverage filter on train rows; imputation and log transform inside the pipeline; target by grouped shift; test sealed until one scoring pass
- **Baseline gate:** a worse-than-constant model cannot ship — enforced before artifacts are written, exits non-zero
- **Real expanding-window CV** with committed result tables (`models/cv_results_*.csv`)
- **Pre-registered refit** with train+val reported as sensitivity (MAE 2.03, bias −0.86 pp) — decided before test was read
- **Significance testing** on the headline comparison, reported whatever it says
- **Provenance:** panel SHA-256, git commit, library versions, split target-years; bit-deterministic re-run
- **86 passing tests**, including adversarial guards: corrupt the test split and selection stays byte-identical; force a gate failure and no artifacts appear
- **Executed notebooks**, no hardcoded paths, loading the deployed artifact rather than re-selecting

---

## Slide 10: Close (15 seconds)

1. **A defensible null beats an undefended win.** The protocol is the contribution.
2. **Gate your models** — "does it beat the baseline it replaces?" belongs in the build, not the discussion section.
3. **Generate every document number from artifacts** so the report cannot drift from the model.

- **GitHub:** this repository — code, artifacts, tests, executed notebooks, report + PDF
- **App:** verified locally via `streamlit run app.py`; no public URL in this submission (README → Deployment)
- **Report:** `reports/capstone_report.pdf`
- **Contact:** [your email]

---

## Q&A talking points *(not counted in runtime)*

- **Why HGB over Ridge?** CV-best HGB beat CV-best Ridge on validation MAE, 3.89 vs 4.00, and cleared the gate.
- **Is parity a failure?** No. It is a quantified, tested statement about predictability, produced by a protocol with no way to fake it.
- **Why did early stopping matter?** The previous cycle overfit because sklearn's `early_stopping="auto"` is inert below 10,000 samples and n=905. Explicit `True` bounds it — the deployed model runs 45 of 200 iterations.
- **Why pre-register the refit?** Validation contains the COVID crash. Refitting across it biases test predictions: MAE 2.03 and −0.86 pp bias, versus 1.82 and +0.08 pp for train-only.
- **Why not tune harder?** Test-set fishing is exactly how the previous cycle shipped a bad model. The grid is compact, pre-registered, and scored only inside training years.
- **How often retrain?** Annually with WDI updates. Gate and significance must pass each time.
- **Can this inform policy?** As screening context only. Causal claims need experimental or quasi-experimental designs.
- **What would change your mind about the null?** Higher-frequency features, or a test set outside a pandemic-recovery window. Both are in Recommendations.

---

## Appendix *(only if asked)*

- A1: Expanding-window CV tables (`models/cv_results_*.csv`)
- A2: Bootstrap CIs and paired significance (report §7)
- A3: Worst-error rows, all verified against `test_predictions.parquet` (report §7)
- A4: Feature-selection audit — coverage filter, `SE.SEC.ENRR` dropped at 59.9%
