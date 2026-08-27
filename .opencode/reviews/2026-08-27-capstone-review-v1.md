# Senior DS/MLE Review — Africa Growth Explorer Capstone

**Reviewer lens:** Senior Data Scientist & ML Engineer
**Evaluated against:** (1) AnalystLab Internship Brief (baseline), (2) FlyRank Enhanced Plan (`CAPSTONE_SPEC.md` + `.opencode/plans/2026-08-26-africa-growth-ml-capstone.md`)
**Evidence base:** repository at commit `1ceea4c`, all artifacts re-executed in a clean Python 3.11 venv
**Scope note:** presentation visuals excluded at author's request; `presentation/slides_outline.md` reviewed only where it restates metrics.

---

## Verdict at a glance

| Area | Rating | One-line justification |
|---|---|---|
| Methodological rigor — data handling | **Partially Meets** | Panel construction and leakage controls are correct; duplicate policy and country list are not |
| Methodological rigor — feature engineering | **Meets** | Target shift, train-only coverage filter, in-pipeline transforms all correct |
| Methodological rigor — validation design | **Partially Meets** | Temporal split correct in `src/`; notebook selects the winner on the **test set** |
| Methodological rigor — model selection | **Does Not Meet** | Deployed model loses to a constant predictor by 86% on MAE; no baseline gate exists |
| Methodological rigor — metrics | **Partially Meets** | Correct metric set, but directional accuracy is degenerate and reported as a strength |
| Code quality & engineering standards | **Meets** | Clean modular `src/`, type hints, docstrings, zero `print()`, consistent logging |
| Experimentation discipline | **Does Not Meet** | Claimed hyperparameter search does not exist in code; no experiment tracking |
| Reproducibility | **Partially Meets** | Pipeline is bit-level deterministic; notebooks hardcode `C:\dev\...`, README run steps are broken |
| System design & architecture | **Meets** | Artifact-driven serving, metadata as feature contract, precomputed inference assets |
| Production readiness | **Partially Meets** | App boots and predicts reliably; guardrails calibrated on the wrong distribution, no model versioning |
| Reporting integrity | **Does Not Meet** | Verified fabrication across EDA stats, error analysis, CIs, and methodology |

**Overall:** This is a well-engineered delivery vehicle wrapped around an unvalidated model and a report whose numbers do not match the artifacts. The engineering would pass a professional review. The science and the write-up would not.

---

## How I verified

```
python3.11 -m venv .venv-review    # clean env, requirements.txt pins
pytest -q                          →  33 passed
python scripts/finalize_model.py   →  reran end-to-end, artifacts byte-reproduced
streamlit run app.py               →  HTTP 200, all 4 pages, live prediction confirmed
```
Every numeric claim below was recomputed from `data/processed/model_data.parquet`,
`models/*.parquet`, and `models/model_metadata.json`.

---

## What is genuinely strong

Credit where it is due — these are not trivial and several are things most interns miss.

- **Leakage controls are real, not cosmetic.** Median imputation lives inside the sklearn `Pipeline`, so it is fitted on training folds only. Coverage-based feature selection is computed under `train_mask` (verified: `SE.SEC.ENRR` correctly dropped at 59.9% vs the 60% threshold). Target is built with `groupby("iso3").shift(-1)`. `create_temporal_split` produces clean non-overlapping year boundaries.
- **The deterministic re-run is a real reproducibility win.** Re-executing `finalize_model.py` reproduced `model_metadata.json` to float precision (single last-digit difference at 1e-16). Seeds are fixed and honoured.
- **Fair-comparison subsetting (B10)** — dropping the 8 test rows lacking current-year growth so *all* models score on identical observations — is a genuinely sophisticated touch.
- **The causal-interpretation discipline is the best part of this submission.** Disclaimers appear in the app (Overview + Scenario), README, and report §10, with a correct conditional-prediction-vs-counterfactual distinction and a sensible list of methods required for causal claims. This directly satisfies `CAPSTONE_SPEC.md` §2 and exceeds the AnalystLab brief.
- **Artifact-driven serving architecture is correct.** The app loads a serialized pipeline plus precomputed predictions/importance parquet; it never retrains, makes no network calls, and uses `@st.cache_resource` / `@st.cache_data` per spec §15.
- **AGENTS.md compliance on code standards:** zero `print()` across `src/`, `app.py`, `scripts/`; module + function docstrings everywhere; type hints on all signatures; one `tests/test_<module>.py` per `src/*.py`.
- **The app is honest in-product.** The Model Performance page explicitly states the negative R² and that the model does not beat the global mean. That candour is rare and valuable.

---

## Critical findings

### C1 — The deployed model is materially worse than a constant predictor
**Does Not Meet** · *methodological rigor, model selection*

Recomputed from the shipped artifacts:

| Model | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|
| Global mean baseline | **1.90** | **2.84** | **-0.00** |
| Persistence baseline | 2.23 | 4.52 | -1.54 |
| **HGB (deployed)** | **3.54** | 5.00 | **-2.10** |

The shipped decision-support engine is **86% worse on MAE** than predicting the training mean for every country-year, and its R² of -2.10 means it explains *less* variance than a horizontal line by a wide margin.

The root cause is a missing gate. `scripts/finalize_model.py` computes baselines **only on the test set, after the winner is already chosen** (lines ~115-130). Validation baselines are never computed, so nothing in the selection path could have caught this. When I compute them:

```
validation:  global mean MAE 4.038  |  Ridge 3.983  |  HGB 3.906
```

HGB's validation margin over the global mean is 3.3% on n=150 — inside noise — and that razor-thin edge is what promoted it to production. Test then contradicted it decisively.

**Why this matters beyond the number:** the entire Scenario Explorer, the feature-importance narrative, and every policy recommendation in the report are downstream of a model with no demonstrated skill.

### C2 — The evaluation notebook selects the winner on the test set
**Does Not Meet** · *validation discipline, leakage*

`notebooks/02_model_evaluation.ipynb`, cell 13:

```python
if hgb_test_metrics["mae"] <= ridge_test_metrics["mae"]:   # <-- TEST metrics
    winner_name = "HistGradientBoostingRegressor"
```

Two separate failures:

1. **Leakage into model selection.** `CAPSTONE_SPEC.md` §9 is explicit: *"The test set must remain untouched until features are finalized, models are selected, hyperparameters are selected."* Selecting on test violates the project's own stated protocol.
2. **The notebook and the deployment disagree.** The notebook prints `Winner: Ridge` (test MAE 2.46). The deployed artifact is HGB (test MAE 3.54). Every downstream notebook output — the actual-vs-predicted scatter, the residual plot, the bootstrap CIs, the worst-error table, the by-year metrics — describes **Ridge, a model that is not deployed**. The report then mixes numbers from both sources into one table.

Ridge at 2.46 MAE is meaningfully better than HGB at 3.54 and closer to the baseline, so the selection bug also cost real accuracy.

### C3 — The report contains fabricated statistics
**Does Not Meet** · *reporting integrity*

Every figure below was recomputed from the committed panel. These are not rounding differences.

**EDA section (§5):**

| Report claim | Actual | Delta |
|---|---|---|
| GDP growth range −60% to +35% | −49.1 to **+91.8** | max off by 57pp |
| Electricity 5–100%, median ~55% | 0.8–100, median **42.7** | median off by 12pp |
| Internet 0–75%, median ~15% | 0–**91.2**, median **7.1** | median off 2× |
| Inflation −10% to +400% | −16.9 to **+557.2** | max off by 157pp |
| "Worst coverage: FDI, Domestic Credit (~14%)" | FDI **96.5%**, Credit **87.3%** | **inverted** — these are among the best-covered |
| Elec ↔ Internet corr 0.78 | **0.66** | |
| GDPpc ↔ Life exp corr 0.65 | **0.45** | |
| Inflation ↔ growth corr −0.31 | **−0.09** | 3.4× overstated |
| Avg growth ~4% (2000-2010) | **2.12** | ~2× overstated |

**Worst-errors table (§7)** — none of the three rows correspond to real data:

| Report | Actual (deployed model) |
|---|---|
| "Libya 2021: actual +35%, pred −5%, error 40pp" | Libya 2021: actual **−9.42**, pred −2.71, error 6.7 |
| "Equatorial Guinea 2022: actual −12%" | actual **−9.63** |
| "Zimbabwe 2021: actual +15%" | actual **+4.34** |

The genuine worst error — **Cabo Verde 2021, actual +15.15 vs predicted −15.50, 30.65pp** — does not appear in the report at all.

**Also fabricated:** the temporal-performance table (claims 3.8/3.2/3.6; actual 3.66/3.82/3.14, and directional accuracy 50/56/52 vs actual 48/46/64); bootstrap CIs (claims MAE [3.1,4.0], RMSE [4.4,5.6]; actual [3.01,4.17], [3.83,6.30]); and "Mean residual ≈ 0 (unbiased)" when the actual mean residual is **+2.07pp** — a large systematic bias in the opposite direction of the claim.

### C4 — A hyperparameter search is claimed but no such code exists
**Does Not Meet** · *experimentation discipline*

Report §6 states:

> Compact grid: `max_iter ∈ {500, 1000}`, `learning_rate ∈ {0.05, 0.1}`, `max_depth ∈ {3, 5}`
> Expanding-window validation: train 2000-2010 → val 2011-2012, train 2000-2012 → val 2013-2014, etc.
> Selected: max_iter=1000, lr=0.05, depth=5 (best validation MAE)

```bash
grep -rn "GridSearch\|RandomizedSearch\|TimeSeriesSplit\|expanding\|cross_val\|param_grid" \
     src/ scripts/ app.py notebooks/ tests/
# → zero matches
```

No search was run. The three hyperparameters are hardcoded constants in `config/indicators.yaml` and used once. This also means `CAPSTONE_SPEC.md` §9's *required* expanding-window tuning was silently dropped — and then reported as completed.

### C5 — HGB is severely overfit; the poor result is a tuning artifact, not a data limit
**Does Not Meet** · *model development*

`max_iter=1000` with `early_stopping="auto"`. With n=905 training rows, sklearn's `"auto"` resolves to **disabled** (it only activates above 10,000 samples). Confirmed: `n_iter_ = 1000` — all 1000 boosting rounds ran with no stopping criterion.

```
HGB train MAE 1.36   vs   val MAE 3.91      ← 2.9× gap
```

Reducing capacity recovers most of the loss:

| Config | Test MAE |
|---|---:|
| `max_iter=1000, depth=5` (shipped) | 3.54 |
| `max_iter=200, depth=3` | 2.37 |
| `max_iter=100, depth=2` | **1.94** |

At `depth=2` the model is essentially level with the global-mean baseline. The headline failure in C1 is therefore substantially self-inflicted through untuned capacity, not an inherent property of the problem.

### C6 — COVID contamination of the refit produces a systematic −2.07pp bias
**Partially Meets** · *validation design*

The split is described everywhere as "Validation 2018-2020 / Test 2021+", but those are **feature** years. In target space:

| Split | Feature years | **Target years** | Mean target |
|---|---|---|---:|
| Train | 2000-2017 | 2001-2018 | +1.94 |
| Val | 2018-2020 | **2019-2021** | **−0.45** |
| Test | 2021-2023 | **2022-2024** | +1.87 |

The winner is refit on train+val, so the 2020 COVID crash (target mean **−4.76**) is baked into the deployed model's central tendency. Deployed predictions average **−0.20** against an actual test mean of **+1.87** — a 2.07pp systematic downward bias that directly explains the negative R².

The report's "COVID placement is deliberate" note addresses only the *selection* rationale and never confronts the refit consequence. The feature-year vs target-year distinction is also never stated, making "Test: 2021+" ambiguous throughout the docs.

---

## High-severity findings

### H1 — Permutation importance is noise, presented as directional insight
**Does Not Meet** · *interpretation*

10 of 14 permutation importances are **negative** — permuting the feature *improves* test score. That is the canonical signature of a model with no learned signal. Yet README and report §7 render these as a ranked table with a **"Direction: Positive/Negative"** column and build narrative on top:

> "Unemployment positive association is counterintuitive but may reflect (a) measurement issues, (b) structural transformation..."

Two errors compounded: permutation importance magnitude carries **no directional information** (labelling it Positive/Negative is a category error), and the underlying values are noise around zero. Spec §13 asks for Ridge standardized coefficients precisely to obtain direction — that was never implemented.

Compounding this: importance is computed on **`X_test, y_test`** (`finalize_model.py`), reusing the held-out set for interpretation after it was already consumed for reporting.

### H2 — Ridge `ColumnTransformer` silently permutes feature order
**Partially Meets** · *latent correctness bug*

`build_ridge_pipeline` splits features into `("log", [idx])` and `("pass", [rest])`. `ColumnTransformer` **concatenates in transformer order**, so the log-transformed column is moved to position 0. Demonstrated:

```
input order        : ['A', 'B', 'C']  (C is log-transformed)
post-transform     : [log(C), A, B]
naive coef mapping : {'A': 0.232, 'B': 0.274, 'C': 0.274}   ← WRONG
truth              : {'C_log': 0.232, 'A': 0.274, 'B': 0.274}
```

Currently latent — HGB won, and no coefficient interpretation exists. But spec §13 *requires* Ridge coefficients, and the obvious implementation (`zip(feature_names, coef_)`) would mislabel **every coefficient**. This is a trap waiting for the next person.

### H3 — Scenario guardrails are calibrated on the wrong distribution
**Partially Meets** · *production readiness*

`CAPSTONE_SPEC.md` §14: *"Define observed **training-data** minimum and maximum."*

`app.py` `get_feature_range` / `get_feature_percentiles` are called with the **full panel** (2000-2024, all splits):

| Feature | Full-panel P1–P99 (used) | Train-only P1–P99 (correct) |
|---|---|---|
| Inflation | −2.92 … **92.05** | −3.28 … **49.51** |
| Electricity | 2.97 … 100.00 | 2.62 … 99.90 |

A user can set inflation to 80% — far outside training support — and receive **no extrapolation warning**. The guardrail exists but is calibrated ~85% too permissive on the most volatile indicator.

### H4 — Directional accuracy is degenerate and is reported as a headline strength
**Does Not Meet** · *metrics*

`compute_metrics` uses `mean((y_true >= 0) == (y_pred >= 0))`. **80.7% of test targets are ≥ 0.** Therefore any always-positive constant predictor scores exactly 80.7% by construction.

That is precisely the global-mean baseline's reported "80.7% directional accuracy" — headlined in README, report §7, and the slide deck as a strong result. It measures the class prior, not skill. HGB's 52.7% is *lower* only because it predicts negative 47% of the time.

A skill-based framing (balanced accuracy, or accuracy relative to the majority-class rate) is required before any of these numbers can be cited.

### H5 — 48% of the codebase is untested, including all selection and serving logic
**Partially Meets** · *code quality*

| Component | LOC | Tests |
|---|---:|---|
| `src/*.py` | 1,186 | 33 tests, 1 file per module |
| `scripts/finalize_model.py` | 195 | **none** |
| `app.py` | 913 | **none** |

`AGENTS.md` rule 2 is satisfied on the letter (one test file per `src` module) but the untested 1,108 lines contain *all* winner-selection logic, *all* artifact writing, and *all* serving logic.

The existing 33 tests are shape/smoke assertions over synthetic `np.random` data. None asserts a metric against a known ground-truth value beyond one trivial case, and — critically — **no test would have caught C1, C2, or C5**. A single `assert winner_test_mae < global_mean_test_mae` in `finalize_model.py` would have blocked this release.

---

## Medium-severity findings

| # | Finding | Rating | Evidence |
|---|---|---|---|
| M1 | **README pipeline instructions are broken.** Steps 4-5 document `python -m src.train` and `python -m src.evaluate`. Neither module has a `__main__` block; both exit 0 doing nothing. The real entry point `scripts/finalize_model.py` is **never mentioned in the README**. A reviewer following the README produces no model. | Does Not Meet | verified: silent no-op exit 0 |
| M2 | **Notebooks hardcode `Path(r'C:\dev\africa-growth-ml')`.** Neither notebook runs on any other machine. README's deployment checklist claims "✅ No local absolute paths" — false. Outputs also leak the author's username (`C:\Users\ingex\...`). | Does Not Meet | `notebooks/*.ipynb` cell 2 |
| M3 | **Country list is incomplete and misdescribed.** Config has **53** ISO3 codes (plan and notebook markdown both say 54). ESH is absent from WDI → correctly warned. But **Mauritius (MUS) and Sudan (SDN) are missing from the config entirely** — never warned, silently absent. Final panel = 52 countries, presented as complete UN coverage. Sudan is a material omission for African growth analysis. | Partially Meets | recomputed vs UN member list |
| M4 | **Logging bug raises `TypeError` on every bootstrap call.** `src/evaluate.py:95` uses `"Bootstrap CI (%%.0f%%): ..."` — the escaped `%%` consumes no argument, so 4 args are passed to a 3-placeholder string. Python's logging swallows it as `--- Logging error ---` on stderr. Violates AGENTS.md rules 4 and 5. Test asserts only `lower < upper`, so it was never caught. | Does Not Meet | reproduced |
| M5 | **`pyproject.toml` packaging is broken.** `packages.find.where = ["src"]` but there is no package *under* `src/` — `src/__init__.py` makes `src` itself the package. `find_packages(where="src")` → `[]`. `pip install -e .` installs nothing importable as `src.*`, yet AGENTS.md instructs exactly that for notebook setup. The notebooks paper over it with the hardcoded Windows path. | Does Not Meet | `find_packages` verified |
| M6 | **Spec §7 duplicate policy not implemented.** Spec requires checking `(iso3, year)` duplicates and *"investigate conflicting duplicate records rather than silently dropping them."* `pivot_to_country_year` uses `aggfunc="first"`, which silently collapses conflicts — the exact behaviour the spec prohibits. No duplicate check exists anywhere. | Does Not Meet | `src/data.py` |
| M7 | **No model versioning or data provenance.** `model_metadata.json` lacks: training timestamp, WDI vintage/download date, sklearn version, row counts, split sizes. The 1.1MB joblib is committed with no hash manifest. Artifact-to-data lineage is unverifiable. | Partially Meets | metadata inspected |
| M8 | **No PDF report.** AnalystLab deliverable #1 is explicitly *"Project Report (PDF)"*. Only `reports/capstone_report.md` exists. | Does Not Meet | `ls reports/` |
| M9 | **No deployment link, no dashboard screenshot.** Spec §20 items 12 and 13. README has a generic Streamlit badge, not a live URL. Deployment is therefore unverified end-to-end on Streamlit Cloud. | Does Not Meet | README |
| M10 | **`.streamlit/config.toml` sets `enableCORS = false` and a hardcoded `port = 8501`.** Streamlit emits a security warning on boot; the port directive conflicts with Cloud's managed port. | Partially Meets | captured in app boot log |
| M11 | `requirements.txt` ships `pytest` into the production runtime. | Partially Meets | line 10 |
| M12 | Dead code: `filter_african_countries`'s `aggregate_codes` set (with duplicate entries) is unreachable — the explicit ISO3 list already excludes aggregates. `clean_numeric`'s placeholder handling never fires because `pivot_table` has already coerced to float. Both signal copy-paste rather than reasoning. | Partially Meets | `src/data.py` |

---

## Alignment: AnalystLab Internship Brief

| Brief requirement | Status | Notes |
|---|---|---|
| Step 1 — Problem definition | **Meets** | Clear target, horizon, scope, users, impact; strong decision-support framing |
| Step 2 — Data collection & understanding | **Meets** | WDI-only, real bulk CSV (396,970 × 70), dtypes and structure profiled in nb 01 |
| Step 3 — Cleaning & preprocessing | **Partially Meets** | Missingness and numeric conversion handled well; **duplicates never checked** (M6); outliers inspected but retained per spec — correct |
| Step 4 — EDA | **Partially Meets** | Notebook 01 is genuinely good (executed, coverage analysis, missingness heatmap, distributions, IQR outliers, correlations). But the **report's EDA prose contradicts the notebook's own outputs** (C3) |
| Step 5 — ≥2 ML models | **Meets** | Ridge + HistGradientBoosting, both real pipelines |
| Step 5 — "select the best-performing model" | **Does Not Meet** | Selected model loses to a constant (C1); notebook selects on test (C2) |
| Step 6 — Evaluation metrics | **Partially Meets** | MAE/RMSE/R²/directional all present; directional metric degenerate (H4) |
| Step 6 — "Explain what the results mean" | **Does Not Meet** | Explanation rests on fabricated numbers (C3) and noise-based importance (H1) |
| Step 7 — Insights & recommendations | **Partially Meets** | Recommendations are appropriately cautious and well-written, but are downstream of an unvalidated model |
| Step 8 — Deployment | **Partially Meets** | Streamlit app works locally (verified HTTP 200, live prediction); no public link (M9) |
| Deliverable — Report PDF | **Does Not Meet** | Markdown only (M8) |
| Deliverable — Source code | **Meets** | Notebooks + `src/` modules, clean structure |
| Deliverable — GitHub repo w/ README + data link | **Meets** | README is thorough; WDI source and download steps documented |
| Deliverable — Presentation | **Meets** (content) | Outline complete; visuals out of scope per your note — but it inherits the C3 numbers |

**Baseline-brief read:** the brief is generously scoped and this submission clears most of it on structure. It fails specifically on the two places the brief asks for *judgement* rather than *activity*: selecting the best model, and explaining what the results mean.

---

## Alignment: FlyRank Enhanced Plan

### The plan's own fix list — did it land?

| ID | Intent | Landed? | Evidence |
|---|---|---|---|
| B1 | `models/`, `data/processed/` not gitignored | ✅ | `git check-ignore` returns nothing |
| B2 | Persistence = current-year growth | ✅ | correct implementation + test |
| B3 | Feature contract from `model_metadata.json` | ✅ | app reads metadata, not config |
| B4 | Slider full range, P1-P99 warning | ⚠️ | implemented, but on the **full panel** not training data (H3) |
| B5 | Metadata actually written | ✅ | `finalize_model.py` writes it |
| B6 | Test syntax errors fixed | ✅ | 33 tests pass |
| B7 | MENA substring trap | ✅ | explicit ISO3 list + regression test — **but list is incomplete** (M3) |
| B8 | Picklable `clip_log1p` | ✅ | verified: Ridge pickle embeds `src.features`; correct by construction |
| B9 | Slider default clamp | ⚠️ | clamp absent from `app.py`; safe only because range derives from the same panel |
| B10 | Fair eval subset | ✅ | 8 rows dropped, all models on identical observations |
| B11 | Precomputed predictions/importance | ✅ | parquet artifacts, app reads them |
| G1-G7 | Report, presentation, `__main__`, pytest, LICENSE, country metadata, bootstrap CI | ✅ mostly | report is `.md` not `.pdf`; `__main__` added to `data`/`features` only — README documents `train`/`evaluate` which lack it (M1) |

**The plan executed its own checklist competently.** ~9 of 11 blockers fully landed.

### Where the enhanced plan improved on the baseline

Genuine additions beyond the AnalystLab brief, all delivered: temporal validation with an explicit leakage rationale; two real baselines; a serialized single-pipeline artifact with a metadata feature contract; precomputed inference assets for cold-start latency; scenario guardrails; a rigorous causal-interpretation boundary; centralized YAML config; per-module unit tests; project visual language.

### Where the enhanced plan deviates from — or fails — the baseline

1. **The plan has no model-quality gate.** This is the deepest structural flaw. Across 20 tasks and 3,856 lines, *no task asks whether the model is any good.* Task 2.3 selects a winner by validation MAE and ships it unconditionally. The plan optimizes deployment reliability and repo hygiene while leaving the scientific core entirely ungoverned — which is exactly how C1 shipped.
2. **The plan silently dropped a mandatory spec requirement.** `CAPSTONE_SPEC.md` §9 requires expanding-window hyperparameter tuning. No plan task implements it; Tasks 2.1/2.3 hardcode config constants. The report then claims it was performed (C4). A dropped requirement became a false claim.
3. **The plan's "notebooks must be executed" rule produced executed-but-inconsistent artifacts.** It mandated real outputs (good) but never required the notebook's conclusions to agree with the deployed artifact — so nb 02 ships a `Winner: Ridge` verdict against a deployed HGB (C2).
4. **The plan's AGENTS.md compliance table is self-certifying.** Rule 6 ("No leakage") is marked satisfied by the temporal split, yet the plan's own notebook task introduced test-set selection. Rule 1 ("never claim code works without running it") is marked satisfied, yet the report asserts a hyperparameter search that was never run.
5. **B4's guardrail spec was weakened in implementation** from training-data bounds to full-panel bounds (H3), silently loosening the safety property the plan set out to add.

**Enhanced-plan read:** the plan is a strong *engineering* plan and a weak *scientific* one. It meaningfully improves portfolio-readiness on architecture, deployment, and responsible-use framing. It does not improve — and in the notebook case actively degrades — methodological validity.

---

## Risk register

| Risk | Severity | Impact |
|---|---|---|
| A model with negative skill is presented as a decision-support tool for development finance | **Critical** | Reputational; a reviewer who checks the baseline column will see the model loses to a constant |
| Report numbers do not match artifacts | **Critical** | Any spot-check destroys credibility of the whole submission; in a professional setting this is a integrity finding, not a quality finding |
| Test set consumed by selection *and* interpretation *and* reporting | **High** | No untouched held-out estimate remains; all reported generalization numbers are optimistically biased |
| Extrapolation guardrail ~85% too permissive on inflation | **High** | Users get unwarned nonsense predictions in exactly the regime the guardrail was built for |
| Notebooks non-executable off the author's machine | **Medium** | Reproducibility claim fails on first attempt by any reviewer |
| README pipeline steps produce no model | **Medium** | Clean-clone reproduction fails |
| No versioning/provenance on committed artifacts | **Medium** | Cannot tie model to WDI vintage; silent staleness |

---

## Prioritized remediation

**Before this is shown to anyone (P0)**

1. **Add a hard baseline gate** in `finalize_model.py`: compute global-mean and persistence on **validation**, and refuse to write artifacts unless the winner beats both. Fail loudly.
2. **Fix the model.** Set `early_stopping=True` with `validation_fraction` + `n_iter_no_change`, and reduce capacity — `max_iter≈100-200, max_depth=2-3` already reaches test MAE 1.94. Then honestly report whether it beats the baseline; "a well-tuned model ties the mean" is a perfectly respectable capstone finding.
3. **Correct or delete every fabricated number** in `reports/capstone_report.md`. Generate the EDA stats, worst-error table, yearly metrics, and CIs programmatically from the artifacts — never by hand. Remove the hyperparameter-search paragraph or implement the search.
4. **Fix notebook 02** to select on validation and to load and evaluate the *deployed* artifact, so notebook, metadata, README, and report tell one story.

**Before submission (P1)**

5. Reframe directional accuracy against the 80.7% majority-class rate, or replace with balanced accuracy.
6. Repoint `get_feature_range` / `get_feature_percentiles` at `data[data.year <= train_end]`.
7. Add `MUS` and `SDN` to `config/indicators.yaml`; make the count assertion explicit in `tests/test_config.py`.
8. Fix `src/evaluate.py:95` format string; add a `caplog` assertion so it cannot regress.
9. Parameterize notebook roots (`PROJECT_ROOT = Path.cwd().parents[0]` or an env var) and re-execute both.
10. Fix README pipeline steps to reference `scripts/finalize_model.py`; either add `__main__` to `src/train.py` and `src/evaluate.py` or stop documenting them.
11. Export the report to PDF; add the live Streamlit URL and a dashboard screenshot.

**Portfolio hardening (P2)**

12. Add tests for `finalize_model.py` (winner selection, metadata contract, baseline gate) and a smoke test that imports `app.py` and exercises `prepare_scenario_input` + `predict`.
13. Fix `pyproject.toml` to `packages = ["src"]` with `where = ["."]`.
14. Stamp `model_metadata.json` with training timestamp, WDI vintage, sklearn version, and split row counts.
15. Add the Ridge coefficient view — and when you do, extract names via `ColumnTransformer.get_feature_names_out()`, never `zip(feature_names, coef_)` (H2).
16. Drop `pytest` from `requirements.txt`; remove `enableCORS`/`port` from `.streamlit/config.toml`.
17. Add the `(iso3, year)` duplicate check that spec §7 requires.

---

## Closing assessment

The distance between the engineering and the science here is unusually wide.

The repository would survive a professional code review: modular, typed, documented, logged, tested at the module level, deterministic, artifact-driven, with a serving architecture that correctly separates training from inference. The causal-interpretation work is genuinely above the level the brief asks for, and the in-app admission of negative R² shows real intellectual honesty.

But three things undermine it. A model was shipped without anyone checking whether it beat the constant it was benchmarked against. The test set was used for model selection in the notebook, so the one number that should be trustworthy no longer is. And the report describes a project that does not exist — statistics that contradict the committed data, error cases that never occurred, and a hyperparameter search that was never run.

The encouraging part: C1 and C5 are the same fix, and it is a small one — the model is not fundamentally incapable, it is untuned. C2 is a four-line change. C3 is tedious but mechanical, and the fix (generate report numbers from artifacts) permanently prevents recurrence. This is roughly a day of focused work away from being a genuinely strong portfolio piece.

The transferable lesson for the enhanced plan: it treated "does the artifact exist and load?" as the definition of done. For an ML system, done means "does the artifact beat the thing it is supposed to replace?" Adding that single gate to the plan would have caught the most serious finding in this review before a line of the report was written.
