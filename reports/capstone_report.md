# Africa Growth Explorer: Capstone Report

**A Machine Learning Decision-Support System Using World Bank Development Indicators**

*Data Science Capstone · AnalystLab Internship*

> **On the numbers in this report.** Every figure is generated from committed
> model artifacts by `scripts/build_report_assets.py` and pasted from
> `reports/generated/`. `tests/test_report_assets.py` fails the build if any
> document quotes a metric the artifacts do not support. Nothing here is
> hand-computed. The model artifacts carry a provenance block —
> `models/model_metadata.json` records the creation timestamp, git commit,
> library versions, panel SHA-256, split sizes and target-year windows.

---

## 1. Introduction

Economic growth drives poverty reduction and living standards across Africa, and the analysts who allocate development capital need some way to compare countries against expected near-term performance. That is the motivation for this project.

The 54 UN African member states differ enormously in economic structure, resource endowment and development trajectory, but they share infrastructure gaps, human capital constraints and exposure to external shocks. A systematic cross-country comparison can surface patterns that single-country analysis misses.

A bare point prediction — "Ghana's 2024 growth will be 3.2%" — is not very useful on its own. What an analyst needs is the prediction, the indicators moving it, and how it shifts under alternative scenarios. That framing turns a static number into a tool, and it is why this project ships an interactive application rather than a table of forecasts.

The World Bank's World Development Indicators are the natural data source: comprehensive, standardised, cross-country comparable, and already monitored by the policy teams who would use this.

### The headline result, stated first

Knowing a country's development indicators this year does **not** help predict its growth next year — at least not beyond what you would get by guessing the historical average.

In plain terms: the model is no better at forecasting growth than simply guessing the historical average every time.

The detail behind that claim. The model's predictions miss the true growth figure by 1.82 percentage points on average. A trivial rule — ignore every indicator and predict the historical average for every country — misses by 1.90. So the model is ahead by 0.07 percentage points.

That margin is too small to trust. The test set holds only 150 country-years, so the result depends partly on which 150 happened to land there. To measure that, the test set was resampled 5,000 times and the comparison re-run on each resample. Across those resamples the model's margin ranged from **0.04 worse** than the trivial rule to **0.19 better**. Since that range covers outcomes where the model *loses*, the 0.07 lead cannot be separated from chance.

The conclusion is therefore not "the model wins narrowly" but "the model and the trivial rule perform the same". Section 8 sets out why that answer holds up under scrutiny, and Section 12 explains why a well-established negative answer is worth more than a poorly-established positive one.

---

## 2. Problem Statement

### The prediction task

Predict next-year GDP per capita growth (annual %) for African countries from development indicators observed in the current year.

- **Target variable:** GDP per capita growth, annual % (`NY.GDP.PCAP.KD.ZG`)
- **Horizon:** one year ahead — features at *t*, target at *t+1*
- **Worked example:** Ghana's 2019 indicators predict Ghana's 2020 growth

### Geographic scope

An explicit ISO3 list of the 54 UN African member states, plus Western Sahara for completeness, held in `config/indicators.yaml`. The list is explicit rather than derived from a region-name substring, which would sweep in non-African members of "Middle East & North Africa". The committed panel covers 52 of the 54 — see §3.

### Intended users

Development analysts, economic researchers, policy teams, NGOs and students.

### Expected impact

A screening tool for comparing development profiles against modelled growth expectations, with warnings when a user pushes it beyond the range of data it learned from, and a clear statement of what it cannot support.

Since the answer turned out to be negative, the honest description is narrower than "a forecasting tool". What this delivers is a **rigorous test of whether forecasting is possible at all** with this data — and the answer is no. That is a real result, but it is not the same as a working forecaster, and this report does not blur the two.

### Success criterion, fixed in advance

The model had to beat the global-mean, persistence and country-historical-mean baselines **on validation**, with the margin then tested for significance on a sealed test set. A model failing that bar does not ship — this is enforced in code, not by discipline (§4).

---

## 3. Dataset Description

### Source

World Bank **World Development Indicators**, downloaded as `WDI_CSV.zip` from [datatopics.worldbank.org/world-development-indicators](https://datatopics.worldbank.org/world-development-indicators/). The archive contains `WDICSV.csv` (data) and `WDICountry.csv` (metadata). Raw data is not committed (~30 MB); the processed panel is, with its SHA-256 recorded in model metadata.

### Panel shape

52 countries, 1,300 country-year rows, 2000–2024. After target construction 1,205 rows carry a usable next-year target.

> **Known coverage gap.** The committed panel predates the addition of
> Mauritius (MUS) and Sudan (SDN) to the country list, so it covers 52 of the
> 54 member states. The configuration and its tests already encode the full
> list; re-running `python -m src.data && python -m src.features` against a
> fresh `WDI_CSV.zip` produces the complete panel. Western Sahara is listed
> for completeness but WDI publishes no rows for it, so `src.data` logs it as
> missing from the data. This gap is stated rather than rounded away, and
> `tests/test_config.py` carries an `xfail` marking it as a known deviation.

### Indicators

Fourteen candidates across six themes. At panel-construction time the coverage filter examined 15 numeric columns — the 14 candidates plus the raw target column, retained as a current-year-growth feature. `SE.SEC.ENRR` fell below the 60% training-coverage threshold at 59.9% and was dropped, leaving **14 features**: 13 configured indicators plus current-year growth.

### Target construction

```python
target_next_year = panel.groupby("iso3")["NY.GDP.PCAP.KD.ZG"].shift(-1)
```

### Data limitations

- Training-row coverage ranges from 80.8% (government consumption) to 100% (life expectancy, urbanisation, population growth). FDI is 97.0% and domestic credit 87.8% — both among the better-covered series
- No forward-filling of volatile series such as inflation and FDI
- Median imputation happens inside the pipeline, fitted on training folds only
- Regional aggregates (SSF, AFE, AFW, income groups) are excluded by the sovereign-ISO3 list
- `src.data.check_duplicates` inspects `(iso3, year, indicator)` keys. Exact duplicates collapse; conflicting duplicates are logged at WARNING for investigation rather than silently dropped
- The 2020 COVID shock sits in the validation target window by design (§4)

---

## 4. Methodology

### Pipeline

1. Load raw WDI CSV in wide format
2. Filter to African countries via the explicit ISO3 list
3. Select the 14 candidate indicators plus the target code
4. Reshape wide to long and run the duplicate check
5. Pivot to a country-year panel and clean numeric placeholders
6. Create the next-year target via grouped shift
7. Coverage-based feature selection at ≥60%, computed on training rows only
8. Temporal train/validation/test split

### Splits — feature years and target years

Because the target is shifted forward one year, each split's targets sit one year later than its features. This distinction matters: conflating the two conceals a real bias mechanism, so the metadata records both.

| split | feature years | target years | n |
|---|---|---|---|
| train | 2000–2017 | 2001–2018 | 905 |
| val | 2018–2020 | 2019–2021 | 150 |
| test | 2021–2023 | 2022–2024 | 150 |

Validation targets include the 2020 COVID crash. Test targets are the post-pandemic recovery. That mismatch drives the refit decision below.

### Missing data

Features below 60% coverage on training rows are dropped. Rows with a missing target are dropped — this is the last feature year for each country. Remaining gaps are filled by median imputation **inside** the sklearn `Pipeline`, so imputation statistics are fitted on training folds only and never see validation or test data.

### Outliers

Extreme values — hyperinflation, commodity booms and busts, conflict-driven collapses — were inspected in `notebooks/01_data_profiling.ipynb` and retained as legitimate macroeconomic events. No winsorisation, no clipping. GDP per capita is log1p-transformed inside the pipeline.

### Baselines

Three, all computed without future information:

1. **Global mean** — predict the training-period mean for every observation
2. **Persistence** — predict next-year growth as this year's growth
3. **Country historical mean** — predict the country's expanding mean over all years ≤ *t*, falling back to the training global mean where a country has no history

### Models

**Ridge regression.** `SimpleImputer(median) → ColumnTransformer(log1p for GDPpc) → StandardScaler → Ridge(α)`, with α chosen by expanding-window CV over {1, 10, 100, 300, 1000, 3000}.

**HistGradientBoostingRegressor.** `SimpleImputer(median) → HGB`, with `early_stopping=True` set explicitly, `validation_fraction=0.15`, `n_iter_no_change=15`, `l2_regularization=1.0`, and a grid over `max_depth` {2, 3} × `learning_rate` {0.01, 0.03, 0.05} × `max_iter` {100, 200}.

> **Why early stopping is explicit.** sklearn's `early_stopping="auto"`
> disables itself below 10,000 samples. At n=905 an earlier configuration ran
> all 1,000 boosting rounds with no stopping criterion and overfit badly. Set
> explicitly, the deployed model stops at 45 of 200 iterations.

### Hyperparameter tuning

Expanding-window cross-validation with folds strictly inside the training period: train 2000–2010 → validate 2011–2012; train 2000–2012 → validate 2013–2014; train 2000–2014 → validate 2015–2016. Configurations are ranked by mean fold MAE with fold standard deviation as tiebreaker. Neither the validation nor the test split is visible to tuning.

### Refit policy, pre-registered

**The selected model is refit on training data only** (`refit_strategy = "train_only"`). This was decided before test data was read, on regime-mismatch grounds alone: validation targets span 2019–2021 including the COVID crash, while test targets span 2022–2024. Refitting on train+val bakes a depressed central tendency into a model that will be scored on a recovery period. The train+val variant is reported as sensitivity in §7 and was never a candidate for deployment.

### Selection gate

Before any artifact is written, the winner must beat **every validation baseline** on MAE. `enforce_baseline_gate` returns a pass/fail structure; on failure `finalize_model.py` exits non-zero and writes nothing. The only override is `--allow-baseline-failure`, which records the failure in metadata and obliges disclosure. This is what makes a worse-than-a-constant model unshippable by accident rather than by vigilance.

### Metrics

The headline metric is **MAE** — mean absolute error, the average gap between predicted and actual growth, in percentage points. An MAE of 1.82 means predictions are off by 1.82 points in a typical year. Lower is better.

Two supporting metrics:

- **RMSE** — like MAE, but it penalises large misses more heavily. A gap between RMSE and MAE signals that a few big errors dominate.
- **R²** — the share of variation in growth the model explains, where 1.0 is perfect and 0.0 means it does no better than always predicting the average. Negative values mean it does *worse* than that.

**Directional accuracy** — how often the model gets the sign right, growth up or down — is always shown next to two companions, because on its own it flatters any model on this data:

- the **majority-class rate**, meaning how often you would be right by always guessing the more common direction, and
- **directional skill**, the difference between the two. Skill is what matters; accuracy alone is not.

Section 7 shows exactly how misleading the raw figure is here.

Also reported: results broken down by year and country, the largest individual errors, and a significance test described in §7 that asks whether the model's lead over the simple baselines could be chance.

### Test-set discipline

The test split is loaded, scored **once**, and never used for tuning, selection, refit choice or interpretation. Feature attribution is computed on validation. Notebooks load frozen predictions rather than recomputing them.

This is verifiable rather than merely asserted: corrupting the test-period outcomes and re-running the pipeline leaves the CV results, the selected winner and the gate outcome bit-identical, while test MAE moves from 1.82 to 51.40.

---

## 5. Exploratory Data Analysis

All figures are computed from the committed panel by the report-asset generator and re-executed in `notebooks/01_data_profiling.ipynb`.

### Target distribution

GDP per capita growth spans **−49.13 pp to +91.78 pp** across 1,255 observed country-years, with a median of 1.92 pp. **72.8% of observations are non-negative** — the origin of the majority-class rate that makes naive directional accuracy meaningless in §7.

![Feature distributions](../figures/eda_feature_distributions.png)

**Figure 1.** Marginal distributions of all 14 candidate features plus the target. Three shapes matter. Inflation, FDI and GDP per capita are severely right-skewed with long single-sided tails — this is why GDP per capita enters the Ridge pipeline under a log1p transform, and why a scale-invariant tree model is the more natural fit. Electricity access and urbanisation are broad and near-uniform, carrying level information rather than change. The target is sharply peaked near 2 pp with tails in both directions: most country-years are unremarkable, and the variance a model would need to explain sits in a small number of extreme observations.

### Missingness

Coverage is not missing-at-random. The pattern is structural.

![Missingness heatmap](../figures/eda_missingness_heatmap.png)

**Figure 2.** Average missing fraction by country-year. Gaps concentrate in specific country blocks — South Sudan before independence, Ethiopia from 2012, Djibouti and Liberia in the early 2000s — rather than spreading uniformly. Two consequences follow. Median imputation is defensible for scattered gaps but weakest exactly where gaps cluster, so those countries carry wider effective error. And because missingness is country-specific and persistent, a coverage filter computed on the full panel would leak test-period data availability into a training decision. The filter therefore runs on the training mask only.

### Summary statistics

| feature | n | min | median | max | train coverage % |
|---|---|---|---|---|---|
| NY.GDP.PCAP.CD | 1270 | 109.59 | 1060.83 | 19141.51 | 98.29 |
| EG.ELC.ACCS.ZS | 1284 | 0.80 | 42.65 | 100.00 | 98.29 |
| IT.NET.USER.ZS | 1263 | 0.01 | 7.14 | 91.20 | 97.76 |
| FP.CPI.TOTL.ZG | 1192 | -16.86 | 5.03 | 557.20 | 91.45 |
| NY.GDP.PCAP.KD.ZG (target) | 1255 | -49.13 | 1.92 | 91.78 | 96.69 |

Median electricity access is 42.65%, median internet penetration 7.14%, and maximum inflation 557.20% — the Zimbabwe hyperinflation episode. Full table in `reports/generated/table_eda_summary.md`.

### Trends

Mean growth fell across the panel's three macro regimes: **2.12 pp** over 2000–2010, **1.49 pp** over 2011–2019, and **0.67 pp** over 2020–2024. Median 2020 growth was **−3.51 pp**, with a median 2021–2022 recovery of **+1.92 pp**.

### Correlations

| pair | pearson r |
|---|---|
| EG.ELC.ACCS.ZS vs SP.DYN.LE00.IN | 0.71 |
| EG.ELC.ACCS.ZS vs SP.URB.TOTL.IN.ZS | 0.69 |
| EG.ELC.ACCS.ZS vs IT.NET.USER.ZS | 0.66 |
| NY.GDP.PCAP.CD vs EG.ELC.ACCS.ZS | 0.61 |

![Feature correlation matrix](../figures/eda_correlation_matrix.png)

**Figure 3.** Pairwise complete correlations. The dense block in the upper left is the development-level cluster: electricity access, internet users, urbanisation, life expectancy and GDP per capita all correlate at 0.45–0.71. These are five measurements of substantially one latent variable, which is why permutation importance later assigns nearly all credit to a single member of the group (§8) rather than distributing it across five independent signals. The row that matters most is current-year growth, which is essentially uncorrelated with everything else (|r| ≤ 0.13). The most plausible persistence predictor of next-year growth has no linear relationship with any level indicator in the panel.

### What the EDA implies for modelling

1. Development levels are strongly collinear slow-moving variables. They describe where a country *is*, not how fast it is about to change.
2. Macroeconomic flows — inflation, FDI, growth itself — show weak bivariate association with next-year growth (|r| ≤ 0.1).
3. Together these foreshadow the modelling result. Cross-country level differences do not discriminate next-year growth once pooled, and the volatile series are noisy at annual frequency.

---

## 6. Model Development

### Pipelines

**Ridge.** `SimpleImputer(median) → ColumnTransformer(log1p GDPpc | passthrough) → StandardScaler → Ridge(α*)`. Coefficient extraction uses `get_transformed_feature_names` because the ColumnTransformer reorders columns — the log-transformed column moves to position 0, so pairing input names with `coef_` positionally would mislabel 9 of 14 coefficients.

**HGB.** `SimpleImputer(median) → HistGradientBoostingRegressor(max_depth*, learning_rate*, max_iter*, l2=1.0, early_stopping=True)`, with starred values from CV.

### Cross-validation results

The complete ranked grids are committed as `models/cv_results_ridge.csv` and `models/cv_results_hgb.csv`. Top rows:

| config | mean fold MAE | std | folds |
|---|---|---|---|
| HGB: max_depth=2, learning_rate=0.03, max_iter=200 | 3.21 | 0.75 | 3 |
| HGB: max_depth=2, learning_rate=0.03, max_iter=100 | 3.21 | 0.75 | 3 |
| HGB: max_depth=3, learning_rate=0.01, max_iter=100 | 3.22 | 0.76 | 3 |
| Ridge: alpha=3000 | 3.30 | 0.77 | 3 |
| Ridge: alpha=1000 | 3.35 | 0.79 | 3 |
| Ridge: alpha=300 | 3.46 | 0.83 | 3 |

Two things stand out. CV selects a heavily regularised tree at depth 2, and it pins Ridge at α=3000 — the largest value in the grid, meaning the linear model does best when shrunk hardest toward the mean. Both point the same way as the final result: additional capacity only buys overfitting.

### Reproducibility

`random_state=42` is fixed throughout. The pipeline, CV and bootstrap re-run deterministically — verified by comparing `model_metadata.json` across two independent runs, which differ only in timestamp and git commit. Provenance in metadata covers `created_utc`, `git_commit`, `library_versions` (Python 3.11.2, scikit-learn 1.9.0, pandas 2.3.3, numpy 1.26.4) and `panel_sha256`.

---

## 7. Evaluation Results

### Selection evidence and the gate

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

Selection happened on validation, before any artifact was written: HGB at 3.89 beat the global-mean baseline at 4.04 by 3.75%, and persistence at 5.02 by 22.58%. Had it failed, the build would have exited 2 with no artifacts. Note that a 3.75% margin at n=150 is itself within noise — the null is already visible at the selection stage.

### Statistical significance

The model beats the predict-the-average baseline by 0.07 percentage points. The question is whether that lead is real or luck.

The test set contains 150 country-years. Had a different 150 landed in it, the margin would have come out differently. To measure how much differently, the test set was resampled with replacement 5,000 times — drawing 150 rows at random each time, allowing repeats — and the model-versus-baseline comparison was re-run on every resample. Both were always scored on the same rows, so the comparison stays fair.

That produces 5,000 versions of the margin. The middle 95% of them fall between **0.04 worse** and **0.19 better**:

> **The model's lead over predicting the average: 0.07 percentage points.**
> **Plausible range across resamples: −0.04 to +0.19.**
> **That range includes zero and negative values, so the lead is not reliable.**

If the model genuinely carried signal, nearly every resample would show it ahead. Instead a meaningful share show it behind. The honest reading is that the model and the trivial rule perform the same.

Stated without hedging: on genuinely unseen post-pandemic years, a tuned gradient-boosting model over 14 WDI indicators cannot beat "always predict 1.6% growth".

### Why the 80.7% figure is not the good news it looks like

The model predicts the correct direction — growth positive or negative — 80.7% of the time. In isolation that sounds strong. It is not.

Growth was positive in 80.7% of test country-years. So a rule that ignores the data entirely and always says "growth will be positive" also scores 80.7%. The model has matched a rule that requires no model.

The useful measure is the gap between the two, and here it is **exactly zero**. A second check tells the same story: scoring the up-years and down-years separately and averaging gives 51.3%, which is a coin flip.

Quoting 80.7% on its own would misrepresent the result, so it never appears in this project without the comparison beside it.

### Fit quality on the test set (n=150)

![Actual vs predicted, test set](../figures/modeling_actual_vs_predicted.png)

**Figure 4.** Actual against predicted next-year growth on the sealed test set — the clearest single picture of the null result. Real signal would pull points toward the diagonal. Instead they form a horizontal band around 1.3–2.1 pp across an actual range spanning −10 pp to +15 pp. For the two country-years near −9 pp actual the model predicts roughly +1.4 and +3.7 pp; for the +15 pp observation it predicts +2.6 pp. This compression is not a tuning failure. It is what a correctly regularised model does when the features carry no conditional information, and it is preferable to a model that manufactures confident wrong answers at the tails.

![Residuals vs predicted, test set](../figures/modeling_residuals.png)

**Figure 5.** Residuals against predicted values. The cloud centres on zero with no visible slope, confirming near-zero bias. Vertical spread runs to roughly ±5 pp for typical observations and beyond ±10 pp at the extremes — that is the honest error magnitude. Because predictions occupy such a narrow horizontal range, residual variance is driven almost entirely by variation in the actual outcome rather than by anything the model imposed.

**Average error direction: +0.08 pp** — the model is neither systematically optimistic nor systematically pessimistic, which is what the train-only refit was chosen to achieve.

Resampling the test set 2,000 times puts the model's MAE somewhere between 1.52 and 2.18, and its RMSE between 2.16 and 3.41. The predict-the-average baseline scores 1.90 and 2.84 — both inside those ranges. That is the same finding from another angle: the two are not reliably distinguishable.

### Refit sensitivity

Training the same model on the COVID years as well, rather than stopping before them, pushes test error up to **2.03** and leaves predictions running **0.86 points low** across the board — worse on both counts, exactly as expected. The deployed model therefore stops before COVID. This alternative was ruled out in advance on the reasoning above, not discarded afterwards because the numbers came out badly.

### Performance by year

| feature year | MAE | RMSE | R2 | Dir. acc | Majority rate | Dir. skill |
|---|---|---|---|---|---|---|
| 2021 | 2.06 | 3.45 | 0.07 | 0.84 | 0.82 | 0.02 |
| 2022 | 1.85 | 2.76 | -0.08 | 0.78 | 0.80 | -0.02 |
| 2023 | 1.55 | 1.98 | 0.10 | 0.80 | 0.80 | 0.00 |

No year shows meaningful skill. R² oscillates around zero.

### Worst errors

| country | year | actual | predicted | abs error | context |
|---|---|---|---|---|---|
| Libya | 2021 | −9.42 | 3.72 | 13.15 | civil-war oil collapse |
| Cabo Verde | 2021 | +15.15 | 2.63 | 12.53 | tourism rebound off a −14.9% pandemic year |
| Equatorial Guinea | 2022 | −9.63 | 1.38 | 11.01 | hydrocarbon contraction |
| Seychelles | 2021 | −9.02 | −0.69 | 8.33 | tourism-dependent recovery |
| Libya | 2022 | +8.97 | 1.38 | 7.59 | oil-output volatility |

Every one is a conflict, oil or tourism shock in a small volatile economy — the class of event no lagged annual WDI snapshot anticipates. This is the mechanism behind the null, made concrete.

### Fair comparison

Test rows without a current-year growth value are dropped globally, so the model and the persistence baseline are scored on identical observations.

---

## 8. Interpretation

### Which indicators the model actually uses

To find out how much any single indicator matters, its column is shuffled at random — breaking the link between that indicator and the outcome — and the model is re-scored. If accuracy collapses, the indicator was carrying weight. If nothing changes, it was not. Shuffling 30 times per indicator gives a range rather than a single number, which matters when the effects are small.

The `significant` column answers one question: across those 30 shuffles, was the indicator's effect reliably above zero, or could it just as easily have been nothing? "Noise" means the latter.

This is measured on the validation split, never on the sealed test set.

| feature | name | mean | std | ci_lower | ci_upper | significant |
|---|---|---|---|---|---|---|
| NY.GDP.PCAP.CD | GDP per capita (current US$) | 0.046 | 0.016 | 0.018 | 0.085 | yes |
| SP.POP.GROW | Population growth (annual %) | 0.017 | 0.011 | 0.000 | 0.037 | yes |
| BX.KLT.DINV.WD.GD.ZS | Foreign direct investment (% of GDP) | 0.002 | 0.002 | -0.003 | 0.006 | noise |
| NY.GDP.PCAP.KD.ZG | GDP per capita growth, current year | 0.001 | 0.010 | -0.011 | 0.013 | noise |
| SP.URB.TOTL.IN.ZS | Urban population (% of total) | 0.000 | 0.002 | -0.002 | 0.003 | noise |
| FS.AST.PRVT.GD.ZS | Domestic credit to private sector (% of GDP) | 0.000 | 0.001 | -0.002 | 0.001 | noise |
| SP.DYN.LE00.IN | Life expectancy at birth (years) | 0.000 | 0.001 | -0.002 | 0.001 | noise |
| IT.NET.USER.ZS | Individuals using the Internet (%) | 0.000 | 0.000 | -0.001 | 0.001 | noise |
| FP.CPI.TOTL.ZG | Inflation, consumer prices (annual %) | 0.000 | 0.000 | -0.001 | 0.001 | noise |
| EG.ELC.ACCS.ZS | Access to electricity (%) | 0.000 | 0.000 | -0.000 | 0.001 | noise |
| NE.CON.GOVT.ZS | Government final consumption (% of GDP) | 0.000 | 0.000 | -0.000 | 0.001 | noise |
| NE.TRD.GNFS.ZS | Trade (% of GDP) | 0.000 | 0.000 | -0.001 | 0.001 | noise |
| NE.GDI.TOTL.ZS | Gross capital formation (% of GDP) | 0.000 | 0.000 | -0.001 | 0.001 | noise |
| SL.UEM.TOTL.ZS | Unemployment (% of labour force) | 0.000 | 0.001 | -0.001 | 0.001 | noise |

**Only two of the fourteen indicators have a measurable effect, and both are tiny.** For the other twelve, shuffling the column at random makes no reliable difference to the predictions — the model was not really using them.

There is no dominant predictor here. Any story about "the top drivers of African growth" built from this table would be reading meaning into noise.

![Permutation importance, validation set, CI-significant features only](../figures/modeling_feature_importance.png)

**Figure 6.** The only two indicators whose effect was reliably above zero. The other twelve are left off deliberately — plotting noise invites people to read meaning into it. Note the scale on the horizontal axis. Scrambling the stronger of the two worsens predictions by 0.046 percentage points, against typical errors of about 3.9. That is roughly a one percent effect: real enough to measure, far too small to act on. GDP per capita shows up here because it stands in for the whole cluster of correlated development measures from Figure 3, not because income level drives next year's growth.

### Direction: Ridge standardised coefficients

Fitted on training data at the CV-selected α=3000.

| feature | name | coefficient |
|---|---|---|
| BX.KLT.DINV.WD.GD.ZS | Foreign direct investment (% of GDP) | 0.135 |
| SP.URB.TOTL.IN.ZS | Urban population (% of total) | -0.068 |
| NY.GDP.PCAP.CD_log1p | GDP per capita (log1p) | -0.068 |
| SP.POP.GROW | Population growth (annual %) | -0.050 |
| NE.GDI.TOTL.ZS | Gross capital formation (% of GDP) | 0.041 |
| EG.ELC.ACCS.ZS | Access to electricity (%) | 0.039 |

Every coefficient is 0.14 or smaller, against typical prediction errors of around 3.9 percentage points. Even the straight-line relationships are weak — and the tuning process chose the setting that flattens them hardest toward zero, which is itself a sign there was little to find.

These figures show association, not cause. The negative coefficient on GDP per capita does **not** mean richer countries grow more slowly; it reflects how that one variable behaves once the other thirteen — several measuring nearly the same thing — are already in the model.

### Why this is a finding rather than a shrug

**It is robust.** The null appears on validation (a 3.75% gating margin at n=150, within noise), on test (paired CI spanning zero), under both model families, and across all three test target years.

**It is mechanistically intelligible.** WDI aggregates move slowly — they are levels, not flows. Next-year growth is dominated by events with no representation in the feature space, as every worst-error row in §7 demonstrates.

**It matches prior expectations** from the macro-forecasting literature for pooled annual cross-country panels of this size.

**The alternative is what a leaky protocol produces.** An earlier iteration of this same project reported a dominant electricity predictor and shipped a model materially worse than a constant, having selected its winner on the test set. Removing the leaks removed the apparent signal. A protocol that can find an effect but reports none is more credible than one that always finds something.

### Why importance is not causality

- **Confounding.** Electricity access correlates with institutional quality, geography and resource rents.
- **Reverse causality.** Growth funds infrastructure at least as readily as infrastructure drives growth.
- **Omitted variables.** Commodity prices, political stability, trading-partner growth and climate are all absent.
- **Measurement error.** Many WDI figures are estimates rather than direct counts, produced by statistical agencies with limited resources. Noisy inputs make real relationships look weaker than they are.
- **Pooling.** One model across 52 economies imposes homogeneous slopes the data does not support.

The model answers "what does this profile typically go with?", not "what would happen if we changed this?". Moving a slider in the application shows how the prediction shifts for a country described that way — not the effect of a policy that brought the change about.

---

## 9. Decision-Support Application

A four-page Streamlit application, loading committed artifacts only — no retraining, no network calls. It is publicly deployed at [https://africa-growth-ml.streamlit.app/](https://africa-growth-ml.streamlit.app/). Dashboard screenshots from the live app are committed under `assets/`: `project_overview.png`, `explore_africa.png`, `model_performance.png`, `scenario_explorer.png`.

1. **Project Overview** — problem, data, model card, headline metrics with the significance verdict, causal disclaimer
2. **Explore Africa** — country growth trends, indicator charts, regional comparison
3. **Model Performance** — significance banner, test and validation baseline tables, actual-vs-predicted, residuals, CI-gated importance with the noise table, Ridge direction panel, per-year metrics
4. **Scenario Explorer** — what-if analysis with training-window guardrails and one-at-a-time model deltas

**Guardrails.** Slider ranges and P1–P99 warning bands are computed on the training window only. Inflation's warning band tops out near 49.5 rather than the full-panel 92.1, so a user entering 80% inflation is correctly warned they have left the supported region. Out-of-band defaults are clamped, and the clamp is disclosed rather than silent.

**Attribution display.** Each row re-runs the deployed pipeline changing only that indicator, reported as an individual effect in percentage points, with a caption noting that effects need not sum because the model is nonlinear.

**Performance display.** Every directional figure appears with its majority-class rate and skill. The page opens with the paired-CI verdict rather than burying it.

---

## 10. Causal Limitations

Development indicators are tangled up with each other and with causes the model never sees. Electricity access, internet penetration and credit depth all move with governance quality and structural conditions that are nowhere in the data. Growth funds infrastructure at least as plausibly as infrastructure causes growth. Critical determinants — commodity prices, terms of trade, political stability, education quality, climate shocks, global financial conditions — are absent from the feature set. WDI figures are modelled estimates carrying non-trivial error in low-capacity statistical systems. And one pooled model imposes homogeneous relationships across 52 very different economies.

Moving a slider from 70% to 80% electricity access asks the model one narrow question: what does it predict for a country that looks like this? It does **not** answer what would happen if a country actually built that capacity. In reality nothing else stays still while electricity access rises — incomes, urbanisation and institutions all move with it. The slider shows a different country profile, not the consequences of a policy.

Establishing policy effects requires different tools: natural experiments and instrumental variables, difference-in-differences on policy rollouts, structural causal models with explicit DAGs, randomised trials where feasible, or synthetic control methods.

---

## 11. Recommendations

**For anyone using this tool**

1. **Do not allocate funding from these point predictions.** They are statistically indistinguishable from a constant. This is first because it is the recommendation with consequences.
2. **Use it for descriptive comparison** of development profiles via the Explore page. That is what the data supports.
3. **Treat every scenario delta as a model response**, never an intervention effect.

**For the next modelling cycle**

4. **Prioritise higher-frequency signal** — nightlights, port and air-traffic data, mobile-money flows, survey expectations — over additional annual WDI indicators. Parity argues for better inputs, not more models.
5. **Keep the baseline gate and the pre-registered protocol.** That machinery is the reusable asset from this project.
6. **Re-ingest WDI** to restore Mauritius and Sudan before quoting country coverage in any external document.
7. **Watch for regime breaks.** This null is partly a statement about a 25-year window containing three distinct macro regimes.

---

## 12. Conclusion

**What was built.** A complete pipeline from raw data to a working application, with controls at each step to keep future information out of past decisions:

- WDI ingestion with an explicit policy for duplicate records
- Feature selection gated on training-period coverage
- Time-ordered splits, with feature years and target years tracked separately
- Hyperparameter tuning on expanding windows inside the training period
- A gate that blocks any model failing to beat the simple baselines
- A test set opened exactly once
- Artifacts stamped with their data fingerprint, git commit and library versions
- A Streamlit application that reads those artifacts and warns on out-of-range inputs
- Executed notebooks and 86 tests, several of which exist purely to catch the selection process cheating

**What the model achieved.** The best configuration found misses actual growth by **1.82** percentage points on average, against **1.90** for the rule "always predict the historical average". Resampling the test set 5,000 times puts that 0.07-point lead anywhere between 0.04 *behind* and 0.19 ahead. Because the range includes losing, the lead cannot be called real: knowing these 14 indicators does not help predict next year's growth beyond guessing the average.

**Why that counts as a result.** A null produced by a protocol that *could* have found an effect — and that, in an earlier and weaker form, was fooled into claiming one — is a genuine finding. It tells the reader that annual-frequency country-level WDI aggregates are too coarse and slow-moving for short-run growth forecasting, and it specifies what a serious next attempt would need: higher-frequency data, event-aware features, and per-country structure.

**How the application supports decisions.** By making the model's limits the first thing a user sees: the significance verdict, majority-rate-aware directional metrics, CI-gated importance, training-window extrapolation warnings, and scenario deltas labelled as model responses.

**Main limitations.** The model is only tested on future years for countries it already knows, not on new countries. The test set is small at 150 observations. COVID sits immediately before the test window. The panel covers 52 of 54 countries. Gaps are filled with median values. And because 12 configurations were compared, some of the apparent margin at selection time is chance.

**Future work.** Re-ingest WDI at the full 54-country list. Add higher-frequency indicators and event features such as commodity prices and political-instability indices. Ship prediction intervals in the application. Benchmark against a hierarchical or panel-econometric model — an AR panel with country fixed effects shrinking toward the mean is the obvious next competitor, and notably the tuned HGB essentially rediscovers that solution. Consider per-subregion models.

---

## 13. Threats to Validity

1. **Only tested on new years, not new countries.** Every country appears in all three splits, so these results say how well the model handles future years for countries it has already seen. Performance on a country absent from training is untested.
2. **The test set is small.** With 150 observations the range around the headline comparison is wide. A genuine 0.07-point advantage could not be detected at this sample size — but equally, an advantage larger than 0.19 can be ruled out.
3. **COVID sits next to the test window.** Validation covers the crash, test covers the recovery. The refit policy was chosen in advance to handle this, but a longer post-pandemic window would settle it more cleanly.
4. **Twelve configurations were compared.** Try enough options and one wins on validation by luck alone, which is why the validation margin is treated as soft evidence. The test-set comparison is the number relied on, and it includes zero.
5. **Filling gaps with median values discards information.** Which countries fail to report is not random — conflict-affected states report less — so the gaps themselves carry meaning that median filling erases.
6. **Two countries are missing.** The panel lacks Mauritius and Sudan. Since the finding is negative, adding two countries is unlikely to overturn it, but the next data refresh should confirm that.
7. **The World Bank revises its data.** Definitions and historical values change between releases. The exact dataset used here is fingerprinted so this analysis can be reproduced, but a fresh download may not match it.

---

## Appendix: Key metrics

Regenerated from `models/model_metadata.json` via `python scripts/build_report_assets.py`.

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
