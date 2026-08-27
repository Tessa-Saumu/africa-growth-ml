# Africa Growth Explorer: Capstone Report

**A Machine Learning Decision-Support System Using World Bank Development Indicators**

> **Provenance note.** Every numeric result in this report is generated from
> committed artifacts by `scripts/build_report_assets.py` into
> `reports/generated/` (and pasted from there). `tests/test_report_assets.py`
> fails the build if any document quotes an MAE figure that is not present in
> `reports/generated/metrics.json`. No number in this report is hand-computed.
> Model artifacts carry a full provenance block (`models/model_metadata.json`:
> creation timestamp, git commit, library versions, panel SHA-256, split
> sizes and target-year windows).

---

## 1. Introduction

### Economic Growth and Development Context
Economic growth remains the primary engine for poverty reduction and improved living standards across Africa. Understanding the drivers of near-term GDP per capita growth is critical for development analysts, policy makers, and investors seeking to allocate resources effectively.

### Why African Development Analysis Is Important
The 54 UN African member states represent diverse economic structures, resource endowments, and development trajectories. Yet they share common challenges: infrastructure gaps, human capital constraints, and vulnerability to external shocks. A systematic, data-driven approach to comparing development conditions across countries can reveal patterns that single-country analysis misses.

### Why Decision Support Is More Useful Than Isolated Prediction
A point prediction of "Ghana's 2024 growth will be 3.2%" has limited utility. What decision makers need is: *Given Ghana's current profile, what does the model predict, which indicators move that estimate, and how does the estimate change under alternative development scenarios?* This decision-support framing turns a static prediction into an interactive analytical tool.

### Role of World Bank WDI
The World Development Indicators provide the most comprehensive, standardized, cross-country comparable dataset for development analysis. Using WDI ensures reproducibility, transparency, and alignment with the indicators policy makers already monitor.

### Headline result, stated up front
Fourteen WDI indicators observed at year *t* carry **no statistically significant information** about year *t+1* GDP per capita growth beyond the unconditional mean. The final model attains test MAE 1.82 against 1.90 for the global-mean baseline, and the paired 95% confidence interval on that improvement is [−0.04, +0.19] — it includes zero. This is a *defensible null result*, established with a leakage-free protocol and a pre-registered decision rule, and it is the substantive finding this report communicates rather than a defect to be optimized away (§12).

---

## 2. Problem Statement

### Prediction Task
Predict next-year GDP per capita growth (annual %) for African countries using development indicators observed in the current year.

### Target
- **Variable:** GDP per capita growth (annual %)
- **WDI Code:** `NY.GDP.PCAP.KD.ZG`
- **Horizon:** 1 year ahead (target at *t+1* from features at *t*)

### Time Horizon
- Features observed at year *t*
- Target observed at year *t+1*
- Example: Ghana's 2019 indicators → predict 2020 growth

### Geographic Scope
Explicit ISO3 list of the 54 UN African member states (plus Western Sahara for completeness) in `config/indicators.yaml` — not region-substring filtering, which would pull in "Middle East & North Africa" countries. The committed panel currently covers 52 of the 54 member states; see §3.

### Intended User
Development analysts, economic researchers, policy analysts, government planning teams, NGOs, and students.

### Expected Impact
A screening tool to compare countries' development profiles against modeled near-term growth expectations, with explicit guardrails (extrapolation warnings) and an explicit causal-interpretation boundary. Given the null headline result, the tool's demonstrated value is as a *falsification harness* — it shows what a leakage-free protocol concludes about forecastable signal in annual WDI — rather than as a validated forecaster.

---

## 3. Dataset Description

### World Bank WDI Source
- Downloaded from [datatopics.worldbank.org/world-development-indicators](https://datatopics.worldbank.org/world-development-indicators/)
- File: `WDI_CSV.zip` containing `WDICSV.csv` (data) and `WDICountry.csv` (metadata)
- Raw data is not committed (~30 MB); the processed panel **is** committed, with SHA-256 recorded in model metadata (`data_provenance.panel_sha256`)

### Indicators Considered
14 candidate indicators across 6 themes. At panel-construction time the coverage filter saw 15 numeric columns (the 14 candidates plus the raw target column, retained as the current-year-growth feature); `SE.SEC.ENRR` fell below the 60% training-coverage threshold and was dropped, leaving **14 features** for the model: 13 config indicators plus current-year growth.

### Number of Countries and Panel Shape (from provenance block)
52 countries, 1,300 country-year rows, years 2000–2024.

> **Known coverage gap.** The committed panel was generated before Mauritius
> (MUS) and Sudan (SDN) were added to the country list and therefore covers 52
> of the 54 UN African member states. Re-running `python -m src.data &&
> python -m src.features` against a fresh `WDI_CSV.zip` produces the full
> 54-country panel; the configuration and its regression tests already encode
> the complete list. ESH (Western Sahara) is listed for completeness but WDI
> publishes no rows for it, so `src.data` logs it as missing-from-data.

### Final Feature Set
The 14 panel features all clear the ≥60% training-coverage threshold at finalization time (`feature_selection.dropped` in metadata is empty for that stage; the one candidate historically dropped for low coverage, `SE.SEC.ENRR`, was removed when the committed panel was constructed).

### Target Definition
`target_next_year = panel.groupby("iso3")["NY.GDP.PCAP.KD.ZG"].shift(-1)`

### Data Limitations
- Feature coverage (training rows, computed): from 80.8% (government consumption) to 100% (life expectancy, urbanization, population growth); FDI is **97.0%** and domestic credit **87.8%** — among the better-covered series.
- No forward-filling of volatile variables (inflation, FDI)
- Median imputation inside the pipeline (fitted on training folds only)
- Aggregates (SSF, AFE, AFW, income groups) explicitly excluded by the sovereign-ISO3 list
- Duplicate policy (spec §7): `src.data.check_duplicates` inspects `(iso3, year, indicator)` keys; exact duplicates collapse, **conflicting** duplicates are logged at WARNING for investigation rather than silently dropped
- 2020 COVID shock sits in the validation **target** window by design (see §4)

---

## 4. Methodology

### Data Transformation
1. Load raw WDI CSV (wide format)
2. Filter African countries via explicit ISO3 list
3. Select 14 candidate indicators + target code
4. Reshape wide → long; run the duplicate check (spec §7)
5. Pivot to country-year panel; clean numeric placeholders
6. Create next-year target via grouped shift
7. Coverage-based feature selection (≥60%, training rows only)
8. Temporal train/val/test split

### Split Definition — feature years vs target years
Because the target is shifted one year forward, each split's *targets* live one year later than its *features*. Earlier drafts of this project blurred this distinction, which concealed a real bias mechanism; the metadata records both explicitly.

(table from `reports/generated/table_splits.md`)

| split | feature_years | target_years | n |
|---|---|---|---|
| train | 2000–2017 | 2001–2018 | 905 |
| val | 2018–2020 | 2019–2021 | 150 |
| test | 2021–2023 | 2022–2024 | 150 |

### Missing-Data Strategy
1. Drop features with <60% coverage on training rows
2. Drop rows where the target is missing (last feature year per country)
3. Median imputation inside the sklearn `Pipeline` (fitted on training folds only)
4. No future information used at any step (AGENTS.md rule 6)

### Outlier Handling
Extreme values (hyperinflation, commodity booms/busts, conflict collapses) were inspected in `notebooks/01_data_profiling.ipynb` and retained as legitimate macroeconomic events. No winsorization or clipping. GDP per capita is log1p-transformed *inside the pipeline*.

### Baselines (spec §10)
1. **Global mean:** predict the training-period mean for every observation
2. **Persistence:** predict year *t+1* growth = observed year *t* growth
3. **Country historical mean (expanding):** predict the country's mean growth over all years ≤ *t* — no future data; falls back to the training global mean when a country has no history

### Models
1. **Ridge regression:** Imputer → ColumnTransformer(log1p for GDPpc) → StandardScaler → Ridge; regularization α selected by expanding-window CV over {1, 10, 100, 300, 1000, 3000}
2. **HistGradientBoostingRegressor:** Imputer → HGB with **explicit `early_stopping=True`** (validation_fraction 0.15, n_iter_no_change 15), L2=1.0; grid over max_depth {2, 3} × learning_rate {0.01, 0.03, 0.05} × max_iter {100, 200}

> **C5 note.** The previous deployment used `early_stopping="auto"`, which
> sklearn disables below 10,000 samples — with n=905 the model ran all 1000
> boosting rounds with no stopping criterion and overfit the 905-row training
> split. Early stopping is now explicit; the deployed model stops at 45 of 200
> iterations (`deployed_hgb_iters` in metrics.json).

### Hyperparameter Tuning — expanding-window CV (spec §9)
Folds strictly inside the training period: train 2000–2010 → validate 2011–2012; train 2000–2012 → validate 2013–2014; train 2000–2014 → validate 2015–2016. Mean fold MAE ranks configs, with fold std as tiebreaker. No test or held-out-validation data is used by tuning.

### Refit Policy — pre-registered (C6)
**Decision made before evaluating test, on regime-mismatch grounds alone: the selected model is refit on TRAIN ONLY (`refit_strategy = "train_only"`).** Rationale: validation *targets* are 2019–2021 and include the COVID crash (validation target mean ≈ −4.76 in 2020), while test *targets* are 2022–2024 (mean ≈ +1.87). Refitting on train+val bakes that depressed central tendency into the deployed model. The train+val variant is reported as a **sensitivity analysis** (§7), never as the deployed configuration.

### Selection Gate (C1)
Before any artifact is written, the winner must beat **every validation baseline** on MAE (`enforce_baseline_gate`). If it fails, `finalize_model.py` exits non-zero and writes no model artifacts. The only override is `--allow-baseline-failure`, which records the failure in metadata and requires disclosure. This gate is what makes "a model worse than a constant" unshippable by accident.

### Evaluation Metrics (spec §12)
- **Primary:** MAE (percentage points)
- **Secondary:** RMSE, R²
- **Directional:** raw sign-agreement **always reported next to the majority-class rate and skill** (H4 — see §7); balanced directional accuracy is also computed
- **Additional:** metrics by year and country, worst errors, bootstrap CIs, paired bootstrap significance test vs the global-mean baseline

### Test-Set Discipline
The test split is loaded, scored **once** (finalize Step H), and never used for tuning, selection, refit-strategy choice, or interpretation. Feature attribution is computed on validation (H1). Notebooks load the frozen predictions; they do not recompute them.

---

## 5. Exploratory Data Analysis

All figures below are computed from the committed panel by the report-asset
generator (`reports/generated/table_eda_summary.md`, `table_correlations.md`)
and re-executed in `notebooks/01_data_profiling.ipynb`.

### Target distribution
GDP per capita growth spans **−49.13 pp to +91.78 pp** (n=1,255 observed country-years; median 1.92 pp; 72.8% of observations non-negative — the origin of the majority-class rate that makes naive "directional accuracy" meaningless, see §7).

![Feature distributions](../figures/eda_feature_distributions.png)

**Figure 1.** Marginal distributions of all 14 candidate features plus the target. Three shapes matter for modelling. Inflation, FDI and GDP per capita are severely right-skewed with long single-sided tails — this is why GDP per capita enters the Ridge pipeline under a log1p transform and why the tree model, which is scale-invariant, is the more natural fit. Electricity access and urbanisation are broad and near-uniform, carrying level information rather than change. The target itself is sharply peaked at roughly 2 pp with tails in both directions: most country-years are unremarkable, and the variance that a model would need to explain sits in a small number of extreme observations.

### Missingness structure

Coverage is not missing-at-random, and the pattern is structural rather than incidental.

![Missingness heatmap](../figures/eda_missingness_heatmap.png)

**Figure 2.** Average missing fraction by country-year. Gaps concentrate in specific country blocks — South Sudan before independence, Ethiopia from 2012, Djibouti and Liberia in the early 2000s — not uniformly across the panel. Two consequences follow. First, median imputation inside the pipeline is defensible for scattered gaps but weakest exactly where gaps cluster, so the affected countries carry wider effective error. Second, because the missingness is country-specific and persistent, a coverage filter computed on the full panel would leak test-period data availability into a training decision; the filter is therefore computed on the training mask only (§4).

### Summary statistics (computed, full panel)

(table from `reports/generated/table_eda_summary.md`, truncated here to key rows)

| feature | n | min | median | max | train_coverage_pct |
|---|---|---|---|---|---|
| NY.GDP.PCAP.CD | 1270 | 109.59 | 1060.83 | 19141.51 | 98.29 |
| EG.ELC.ACCS.ZS | 1284 | 0.80 | 42.65 | 100.00 | 98.29 |
| IT.NET.USER.ZS | 1263 | 0.01 | 7.14 | 91.20 | 97.76 |
| FP.CPI.TOTL.ZG | 1192 | -16.86 | 5.03 | 557.20 | 91.45 |
| NY.GDP.PCAP.KD.ZG (target) | 1255 | -49.13 | 1.92 | 91.78 | 96.69 |

Key readings: electricity access median **42.65%**, internet penetration median
**7.14%**, inflation max **557.20%** (Zimbabwe-era hyperinflation).

### Trend analysis (computed means)
- Average growth **2000–2010: 2.12 pp**
- **2011–2019: 1.49 pp**
- **2020–2024: 0.67 pp**; median 2020 growth **−3.51 pp** (COVID); median 2021–2022 recovery **+1.92 pp**

### Correlations (computed; |r| ≥ 0.6 pairs)

(table from `reports/generated/table_correlations.md`)

| pair | pearson_r |
|---|---|
| EG.ELC.ACCS.ZS vs SP.DYN.LE00.IN | 0.71 |
| EG.ELC.ACCS.ZS vs SP.URB.TOTL.IN.ZS | 0.69 |
| EG.ELC.ACCS.ZS vs IT.NET.USER.ZS | 0.66 |
| NY.GDP.PCAP.CD vs EG.ELC.ACCS.ZS | 0.61 |

Pairs the narrative previously quoted, now computed: electricity↔internet
**0.66** (was 0.78), GDPpc↔life-expectancy **0.45** (was 0.65),
inflation↔growth **−0.09** (was −0.31, overstated 3.4×),
capital-formation↔growth **0.10**.

![Feature correlation matrix](../figures/eda_correlation_matrix.png)

**Figure 3.** Pairwise complete correlations. The dense red block in the upper-left quadrant is the development-level cluster: electricity access, internet users, urbanisation, life expectancy and GDP per capita all correlate at 0.45–0.71. These are five measurements of substantially the same latent variable, which is why permutation importance later distributes credit almost entirely to one of them (§8) rather than to five independent signals. The row that matters most is `NY.GDP.PCAP.KD.ZG` — current-year growth — which is essentially uncorrelated with everything else (|r| ≤ 0.13 across the board). The strongest predictor of next-year growth in a persistence sense has no linear relationship with any level indicator in the panel.

### Main EDA findings
1. Development *levels* (electricity, internet, urbanization, life expectancy) are strongly collinear slow-moving variables — they carry level information, not year-on-year change.
2. Macroeconomic *flows* (inflation, FDI, growth itself) show weak bivariate association with next-year growth (|r| ≤ 0.1).
3. These two observations already foreshadow the modeling result: cross-country level differences do not discriminate *next-year* growth once pooled, and the volatile series are noisy at annual frequency.

---

## 6. Model Development

### Pipelines
- **Ridge:** `SimpleImputer(median) → ColumnTransformer(log1p GDPpc | passthrough) → StandardScaler → Ridge(α*)`, α* from CV. Coefficient extraction uses `get_transformed_feature_names` because the ColumnTransformer reorders columns (the log column moves to position 0; `zip(features, coef_)` would mislabel every coefficient — H2).
- **HGB:** `SimpleImputer(median) → HistGradientBoostingRegressor(max_depth*, learning_rate*, max_iter*, l2=1.0, early_stopping=True)` with grid-selected *.

### Expanding-window CV — real this time (C4)
Earlier drafts described a hyperparameter search that had never been run. This cycle implements it (`search_hyperparameters` in `src/train.py`); the complete ranked grids are committed as `models/cv_results_ridge.csv` / `models/cv_results_hgb.csv`. Top rows:

(table from `reports/generated/table_cv_results.md`)

| config | mean fold MAE | std | folds |
|---|---|---|---|
| HGB: max_depth=2, learning_rate=0.03, max_iter=200 | 3.21 | 0.75 | 3 |
| HGB: max_depth=2, learning_rate=0.03, max_iter=100 | 3.21 | 0.75 | 3 |
| HGB: max_depth=3, learning_rate=0.01, max_iter=100 | 3.22 | 0.76 | 3 |
| Ridge: alpha=3000 | 3.30 | 0.77 | 3 |
| Ridge: alpha=1000 | 3.35 | 0.79 | 3 |
| Ridge: alpha=300 | 3.46 | 0.83 | 3 |

The CV picks a heavily regularized HGB (depth 2) and a strongly shrunk Ridge (α=3000 at the grid edge, so linear skill is essentially "predict the mean with slightly tuned damping") — consistent with the null finding: more capacity only overfits.

### Reproducibility
- `random_state=42` fixed throughout; pipeline + CV + bootstrap re-run byte-deterministically (verified by double-run comparison of `model_metadata.json`).
- Provenance recorded in metadata: `created_utc`, `git_commit`, `library_versions` (python 3.11.2 / scikit-learn 1.9.0 / pandas 2.3.3 / numpy 1.26.4 in this environment), `panel_sha256`.

---

## 7. Model Evaluation

### Selection evidence (validation) and the baseline gate

| Model | Split | MAE | RMSE | R2 | Dir. acc | Majority rate | Dir. skill |
|---|---|---|---|---|---|---|---|
| Global mean baseline | test | 1.90 | 2.84 | -0.00 | 0.81 | 0.81 | 0.00 |
| Persistence baseline | test | 2.23 | 4.52 | -1.54 | 0.77 | 0.81 | -0.03 |
| Country historical mean baseline | test | 1.94 | 2.88 | -0.03 | 0.78 | 0.81 | -0.03 |
| HistGradientBoostingRegressor (deployed) | test | 1.82 | 2.79 | 0.03 | 0.81 | 0.81 | 0.00 |
| Global mean baseline | validation | 4.04 | 6.14 | -0.18 | 0.51 | 0.51 | 0.00 |
| Persistence baseline | validation | 5.02 | 8.27 | -1.14 | 0.55 | 0.51 | 0.03 |
| Ridge (CV-best) | validation | 4.00 | 6.08 | -0.16 | 0.51 | 0.51 | 0.00 |
| HGB (CV-best) | validation | 3.89 | 5.97 | -0.11 | 0.53 | 0.51 | 0.01 |

(Full table: `reports/generated/table_model_comparison.md`.)

The gate passed **on validation before any artifact was written**: HGB 3.8869 vs global-mean 4.0383 (margin +3.75%) and vs persistence 5.0207 (+22.58%). Had the gate failed, the build would have exited 2 with no artifacts.

### Statistical significance (the number that matters)
The paired bootstrap over test absolute-residual differences (model vs global-mean baseline; 5,000 resamples, seed 42):

> **Paired MAE improvement = +0.074 pp, 95% CI [−0.042, +0.187] — spans zero. Not significant at 95%.**

The model achieves **parity** with the unconditional mean, not a demonstrated victory. Stated positively: on genuinely held-out post-pandemic years, a tuned gradient-boosting ensemble of 14 WDI indicators cannot beat "predict 1.6%" once the mean itself is estimated from training data. That is the capstone's substantive finding.

### Directional accuracy, de-degenerated (H4)
80.67% of test targets are ≥0, so **any** always-positive predictor — including the global-mean baseline — scores 80.67% directional accuracy by construction. Reported next to the majority-class rate, the deployed model's directional **skill is 0.00pp** and its balanced directional accuracy is 51.3%: no sign information beyond the class prior. Earlier drafts presented a raw directional accuracy *below* the majority rate as a strength; that framing is retracted.

### Fit quality (test set, n=150)

![Actual vs predicted, test set](../figures/modeling_actual_vs_predicted.png)

**Figure 4.** Actual versus predicted next-year growth on the sealed test set. This is the clearest single picture of the null result. If the model carried real signal, points would track the diagonal; instead they form a horizontal band at roughly 1.3–2.1 pp across an actual range spanning −10 pp to +15 pp. The model has learned the unconditional mean and little else. Note the behaviour at the extremes: for the two country-years near −9 pp actual, the model predicts approximately +1.4 and +3.7 pp, and for the +15 pp observation it predicts +2.6 pp. The compression is not a tuning failure — it is what a correctly regularised model does when the features carry no conditional information, and it is preferable to a model that manufactures confident wrong answers at the tails.

![Residuals vs predicted, test set](../figures/modeling_residuals.png)

**Figure 5.** Residuals against predicted values. The cloud is centred on zero with no visible slope, confirming the near-zero mean residual reported below and the absence of the systematic bias that the previous train+val refit introduced. The vertical spread — roughly ±5 pp for typical observations and beyond ±10 pp for the extremes — is the honest error magnitude. Because predictions occupy a narrow horizontal range, residual variance is driven almost entirely by variation in the actual outcome rather than by any structure the model imposed.

- **Mean residual (actual − predicted): +0.080 pp** — near-zero bias, as intended under the pre-registered train-only refit. (The previous deployed model carried a −2.07 pp systematic bias from refitting across the COVID regime; the sensitivity analysis below reproduces that mechanism honestly.)
- Bootstrap 95% CIs (2,000 resamples, seed 42): **MAE [1.52, 2.18]**, **RMSE [2.16, 3.41]** — both intervals contain the corresponding global-mean baseline values.

### Refit sensitivity (C6 quantified, not hidden)
Refitting the same selected model on train+val instead of train-only yields test MAE **2.03** with mean prediction−actual bias **−0.86 pp** — worse than the deployed model and systematically depressed, exactly as the pre-registered rationale predicted. The primary result uses train-only refit; this paragraph is documentation of the counterfactual, decided before test was read.

### Performance by feature year (targets one year later)

(table from `reports/generated/table_yearly_metrics.md`; `year` = feature year)

| year | MAE | RMSE | R2 | Dir. acc | Majority rate | Dir. skill |
|---|---|---|---|---|---|---|
| 2021 | 2.06 | 3.45 | 0.07 | 0.84 | 0.82 | 0.02 |
| 2022 | 1.85 | 2.76 | -0.08 | 0.78 | 0.80 | -0.02 |
| 2023 | 1.55 | 1.98 | 0.10 | 0.80 | 0.80 | 0.00 |

No year shows meaningful skill; the R² values oscillate around zero.

### Worst Errors

Top absolute errors (from `reports/generated/table_worst_errors.md`; actual/predicted in pp):

- **Libya 2021**: actual −9.42, predicted 3.72 (error 13.15 pp) — civil-war oil-collapse year
- **Cabo Verde 2021**: actual +15.15, predicted 2.63 (error 12.53 pp) — tourism-rebound base effect after a −14.9% pandemic year
- **Equatorial Guinea 2022**: actual −9.63, predicted 1.38 (error 11.01 pp) — hydrocarbon contraction
- **Seychelles 2021**: actual −9.02, predicted −0.69 (error 8.33 pp)
- **Libya 2022**: actual +8.97, predicted 1.38 (error 7.59 pp)

All five are conflict/oil/tourism-shock years in small, volatile economies — events no lagged annual WDI snapshot anticipates. (An earlier draft of this report contained an error table whose rows did not exist in the data; that table was fabricated and is replaced by this computed one.)

### Fair-comparison note (B10)
Test rows lacking a current-year growth value are dropped globally so the ML model and the persistence baseline are scored on identical observations.

---

## 8. Interpretation

### Magnitude: permutation importance with confidence intervals (validation set)

(table from `reports/generated/table_feature_importance.md`)

| feature | name | importance_mean | importance_std | ci_lower | ci_upper | is_significant |
|---|---|---|---|---|---|---|
| NY.GDP.PCAP.CD | GDP per capita (current US$) | 0.046 | 0.016 | 0.018 | 0.085 | yes |
| SP.POP.GROW | Population growth (annual %) | 0.017 | 0.011 | 0.000 | 0.037 | yes |
| BX.KLT.DINV.WD.GD.ZS | Foreign direct investment, net inflows (% of GDP) | 0.002 | 0.002 | -0.003 | 0.006 | noise |
| NY.GDP.PCAP.KD.ZG | GDP per capita growth (annual %), current year | 0.001 | 0.010 | -0.011 | 0.013 | noise |
| SP.URB.TOTL.IN.ZS | Urban population (% of total population) | 0.000 | 0.002 | -0.002 | 0.003 | noise |
| FS.AST.PRVT.GD.ZS | Domestic credit to private sector (% of GDP) | 0.000 | 0.001 | -0.002 | 0.001 | noise |
| SP.DYN.LE00.IN | Life expectancy at birth, total (years) | 0.000 | 0.001 | -0.002 | 0.001 | noise |
| IT.NET.USER.ZS | Individuals using the Internet (% of population) | 0.000 | 0.000 | -0.001 | 0.001 | noise |
| FP.CPI.TOTL.ZG | Inflation, consumer prices (annual %) | 0.000 | 0.000 | -0.001 | 0.001 | noise |
| EG.ELC.ACCS.ZS | Access to electricity (% of population) | 0.000 | 0.000 | -0.000 | 0.001 | noise |
| NE.CON.GOVT.ZS | General government final consumption expenditure (% of GDP) | 0.000 | 0.000 | -0.000 | 0.001 | noise |
| NE.TRD.GNFS.ZS | Trade (% of GDP) | 0.000 | 0.000 | -0.001 | 0.001 | noise |
| NE.GDI.TOTL.ZS | Gross capital formation (% of GDP) | 0.000 | 0.000 | -0.001 | 0.001 | noise |
| SL.UEM.TOTL.ZS | Unemployment, total (% of total labor force) | 0.000 | 0.001 | -0.001 | 0.001 | noise |

Reading this table honestly: **2 of 14 features are distinguishable from zero** at
95%, both with tiny magnitudes (0.046 and 0.017 mean importance on a metric in
squared-error units per permutation), and 12 of 14 have intervals straddling
zero. There is no dominant feature. A previous version of this project ranked
these same values with a "Direction: Positive/Negative" column and built
narratives on top of them; that was doubly wrong — permutation importance
carries no sign semantics (it measures *degradation from scrambling*), and the
values are noise. Both errors are retracted here.

![Permutation importance, validation set, CI-significant features only](../figures/modeling_feature_importance.png)

**Figure 6.** The only two features whose permutation importance has a 95% confidence interval excluding zero, measured on validation. Twelve of the fourteen features are omitted from this chart because their intervals contain zero — they are indistinguishable from noise, and plotting them would invite exactly the over-reading that earlier drafts committed. Note the axis scale: the larger of the two effects is 0.046 in MAE-degradation units against a model MAE near 3.9 on validation, roughly one percent. These are statistically detectable but economically negligible effects, and GDP per capita's appearance here is best read as the model latching onto the development-level cluster identified in Figure 3, not as evidence that income level drives next-year growth.

### Direction: Ridge standardized coefficients (training fit; CV-best α=3000)

(table from `reports/generated/table_ridge_coefficients.md`, top rows)

| feature | name | coefficient |
|---|---|---|
| BX.KLT.DINV.WD.GD.ZS | Foreign direct investment, net inflows (% of GDP) | 0.135 |
| SP.URB.TOTL.IN.ZS | Urban population (% of total population) | -0.068 |
| NY.GDP.PCAP.CD_log1p | GDP per capita (current US$) (log1p) | -0.068 |
| SP.POP.GROW | Population growth (annual %) | -0.050 |
| NE.GDI.TOTL.ZS | Gross capital formation (% of GDP) | 0.041 |
| EG.ELC.ACCS.ZS | Access to electricity (% of population) | 0.039 |

All |coefficients| are ≤0.14 standardized units on a target with ~3.9pp validation MAE: even the *linear* association structure is faint, and heavy shrinkage (α=3000, the grid's largest value) is what fits best. Direction is reported **as association only**: negative GDPpc coefficient reflects conditional (partial) relationships within this small, collinear feature set — not "richer countries grow slower" as a causal claim.

### Why this is a finding, not a shrug
1. **It is robust.** The null appears on the *validation* split (gating margin 3.75% at n=150, within noise), on *test* (paired CI spans zero), under *both* model families, and across all three test target years.
2. **It is mechanistically intelligible.** WDI growth-year aggregates change slowly (levels, not flows); next-year growth is dominated by events (conflict, commodity, policy shocks) with no representation in the feature space (§7 worst-errors).
3. **It matches the macro-forecasting literature's priors** at annual frequency for a pooled cross-country panel of this size.
4. **The alternative outcome is what a leaky protocol produces:** the previous cycle "found" 52.7% directional accuracy, a "dominant electricity predictor" and a shipped model worse than a constant. Removing the leaks removed the mirage.

### Why feature importance ≠ causality (unchanged from prior drafts; still true)
1. **Confounding:** electricity access correlates with institutions, geography, and oil rents.
2. **Reverse causality:** growth funds infrastructure at least as much as infrastructure drives growth.
3. **Omitted variables:** commodity prices, political stability, partners' growth, climate.
4. **Measurement error:** WDI series are modeled estimates in low-capacity statistical systems; errors-in-variables attenuates associations.
5. **Pooling:** one model across all countries imposes homogeneous slopes the data does not support.

The model is a *conditional predictor*. Scenario-slider movements in the app are conditional prediction deltas — never counterfactual policy effects.

---

## 9. Decision-Support Application

### Streamlit architecture (4 pages)
1. **Project Overview** — problem, data, model, honest headline metrics, causal disclaimer
2. **Explore Africa** — country trends, indicator charts, regional comparison
3. **Model Performance** — baseline comparison (test + validation), significance banner, actual-vs-predicted, residuals, CI-gated feature importance with noise table, Ridge direction panel, by-year metrics
4. **Scenario Explorer** — what-if analysis with **training-window** guardrails and one-at-a-time model deltas

### Guardrail design (H3)
Slider ranges and P1–P99 warning bands are computed on the **training window only** (spec §14): e.g. inflation's warning band is the training P1–P99 (upper ≈ 49.5), not the full-panel 92.1, so a user setting 80% inflation is now warned. Out-of-band defaults are clamped into range and the clamp is disclosed to the user.

### Contribution display (H1)
The former "Approx. Contribution = importance × change" table was dimensionally meaningless and is removed. Each row now re-runs the deployed pipeline changing only that indicator ("Individual effect (pp)"), with a caption that effects need not sum because the model is nonlinear.

### Performance display (H4, significance)
Every directional figure is shown with the majority-class rate and skill alongside it; the page opens with the paired-CI verdict.

---

## 10. Causal Limitations

Development indicators are endogenous; electricity access, internet penetration, and credit depth correlate with governance quality and structural features the model cannot observe. Growth funds infrastructure at least as plausibly as infrastructure causes growth. Critical determinants — commodity prices, terms of trade, political stability, education quality, climate shocks, global financial conditions — are absent. WDI figures are modeled estimates with non-trivial error in low-capacity statistical systems. One pooled model imposes homogeneous relationships across 52 very different economies.

Changing a slider from 70% to 80% electricity access asks the model: *what does it predict for a country-year profiled at 80%?* The real world does not hold other factors constant; this is a conditional prediction, not a counterfactual. Establishing policy effects requires natural experiments / instrumental variables, difference-in-differences on policy rollouts, structural causal models with explicit DAGs, RCTs where feasible, or synthetic-control methods.

---

## 11. Recommendations

1. **Do not allocate funding on the basis of this model's point predictions** — the honest headline (§7) says they are statistically indistinguishable from a constant.
2. Use the dashboard for **descriptive comparison** of development profiles (its Explore page) — that is what the data supports.
3. Treat every "scenario delta" as a *model-response* statistic, never an intervention effect.
4. For actual growth forecasting at annual frequency, prioritize **higher-frequency signal** (nightlights, port/air-traffic data, mobile-money flows, survey expectations) over additional annual WDI indicators.
5. Keep the **baseline gate + pre-registered protocol** from this cycle for any future model iteration; it is the reusable asset.
6. Re-ingest WDI to restore MUS/SDN coverage before quoting country coverage in any external document.
7. Monitor regime breaks (pandemic, commodity, conflict) — this null result is partly a statement about a 25-year window that includes three very different macro regimes.

---

## 12. Conclusion

**What was built:** an end-to-end, leakage-controlled ML decision-support system — WDI ingestion with duplicate policy, coverage-gated feature engineering, temporal splits with explicit feature/target-year accounting, expanding-window hyperparameter tuning, a validation baseline gate, a frozen single-test-observation protocol, provenance-stamped artifacts, an artifact-driven Streamlit app with training-window guardrails, executed notebooks, and a test suite (≥60 tests) that regression-guards the selection protocol itself.

**What the model achieved:** the strongest configuration within a pre-registered, honest search reaches test MAE **1.82** against **1.90** for the global-mean baseline, with a paired 95% CI of **[−0.04, +0.19]** on the improvement. Because that interval includes zero, we conclude the 14 WDI indicators evaluated here carry **no statistically significant information** about next-year GDP per capita growth beyond the unconditional mean.

**Why that is a result:** a defensible null, produced by a protocol that *could have* found an effect (and previously was fooled into claiming one), is a genuine scientific contribution: it tells the reader that annual-frequency, country-level WDI aggregates are too coarse and slow-moving for short-run growth forecasting, and it identifies exactly what a serious next attempt would need (higher-frequency data, event-aware features, per-country structure).

**How the dashboard supports decisions:** by making the model's limits the front page — significance verdict, majority-rate-aware directional metrics, CI-gated importance, training-window extrapolation warnings, and one-at-a-time model-delta scenarios.

**Main limitations:** temporal generalization only; n=150 test observations; COVID regime break adjacent to the test window; 52-of-54 country coverage in the committed panel; median imputation; grid multiple-comparison exposure (§13).

**Future work:** re-ingest WDI at the full 54-country list; add higher-frequency indicators; event/antecedent features (commodity prices, political-instability indices); prediction intervals; a hierarchical or panel-Econometrics benchmark (AR panel with country fixed effects and shrinking memory toward the mean is the obvious next competitor — our tuned HGB essentially *discovers* that solution and lands on it); per-subregion models.

---

## 13. Threats to Validity

1. **Temporal generalization only.** Countries are shared across splits; we estimate "next years for known countries", not "unseen countries".
2. **Small test set.** n=150 country-years → CIs on the headline comparison are wide; a true 0.07pp advantage cannot be resolved at this n, and neither can it be ruled out beyond the stated interval.
3. **Regime break.** The val/test target regimes differ (COVID crash vs post-pandemic); our pre-registered refit policy addresses this but a longer post-COVID test window would be cleaner.
4. **Multiple comparisons.** 12 CV configs + 2 families were compared on validation; selection noise inflates validation margins slightly. The paired bootstrap CI on the *test* metric is the number we lean on, and it spans zero.
5. **Median imputation** erases country-level missingness structure (missing-ness can itself be informative — e.g. conflict states report less).
6. **Coverage gap.** The committed panel lacks MUS/SDN; conclusions are unchanged by construction (they are null), but the next re-ingestion should re-verify.
7. **Survivorship in WDI.** Indicator definitions and historical revisions change; the panel is pinned by SHA-256 provenance, not by a guarantee of vintage stability.

---

## Appendix: Key metrics (regenerated from `models/model_metadata.json`)

```json
{
  "model_type": "HistGradientBoostingRegressor",
  "n_features": 14,
  "refit_strategy": "train_only",
  "gate": {"metric": "mae", "passed": true},
  "metrics": {
    "global_mean_baseline": {"mae": 1.8958, "rmse": 2.8379, "r2": -0.0005},
    "persistence_baseline": {"mae": 2.2287, "rmse": 4.5209, "r2": -1.5392},
    "country_historical_mean_baseline": {"mae": 1.9424, "rmse": 2.8816, "r2": -0.0316},
    "winner_test": {"mae": 1.8217, "rmse": 2.7942, "r2": 0.0300}
  },
  "significance": {
    "paired_mae_improvement_vs_global_mean": 0.074,
    "ci_lower": -0.0422,
    "ci_upper": 0.1868,
    "significant_at_95": false,
    "n_bootstrap": 5000
  }
}
```

*Values in this appendix are a copy of generated `reports/generated/metrics.json`
fields; regenerate with `python scripts/build_report_assets.py`.*
