# Project Evidence — Africa Growth Explorer (`africa-growth-ml`)

Forensic record of what this repository actually demonstrates. Nothing in this file is a code change, a README rewrite, or a claim inferred only because a dependency or filename exists.

Review date: 2026-09-25. Local checkout: shallow clone of `295d665d0b02cf1b4c82307e387014ffe4163980` (`main` / `arena/01a0d930-africa-growth-ml`). Full history, pull requests, and issues were read from GitHub (`Tessa-Saumu/africa-growth-ml`), not from the local `git log`, which contains only that one commit.

---

## Context

**When:**
GitHub repository created 2026-08-26. First commit `929fd2b` (2026-08-26, "Initial project structure"). Model artifact timestamp in `models/model_metadata.json`: `created_utc` `2026-08-27T03:08:40+00:00`, recorded `git_commit` `2920aa9839603008e66fb9efc19e7b1518daac6f`. Last commit on `main`: `295d665` (2026-09-01, "Revise README with updated file links and new section"), author Theresia Saumu. Presentation deck dates itself "Aug 2026". No tags, no GitHub releases.

**Why it existed:**
The documents describe an AnalystLab Data Science Internship capstone. That affiliation is stated in `reports/capstone_report.pdf` ("Data Science Capstone · AnalystLab Internship"), `reports/presentation_deck.pdf` ("Presenter: Theresia Saumu", "Program: AnalystLab Data Science Internship Capstone"), and `README.md` acknowledgments. There is no external brief, supervisor commit, or programme artefact in the repository that independently confirms the internship. The implemented question, which the code actually answers, is whether a fixed set of World Bank World Development Indicators observed in year `t` predicts GDP per capita growth in year `t+1` for African countries, under a temporal split, and whether that prediction is better than trivial rules.

**Solo / team:**
One human GitHub account. `gh` contributors API returns only `Tessa-Saumu` (44 contributions). Commit author names on that account are `Theresia Saumu` and `Tessa-Saumu`; the only local commit email is `theresia.saumu@gmail.com`. No second human committer. Sixteen of 44 commits contain `Co-authored-by: arena-agent <297053741+arena-agent@users.noreply.github.com>`. Pull request #3 was opened by `app/arena-ai-coding-agent`. Acknowledgments also name the World Bank (data), scikit-learn, Streamlit, and "the AnalystLab internship programme for project guidance." No supervisor or teammate appears in git.

**My role:**
Cannot be proven beyond the git identity. Every commit is attributed to the `Tessa-Saumu` / Theresia Saumu account, including four merge commits. The README "Development process" section is a first-person self-report: the author chose the project, framing, standards, scope, and modelling decisions, directed the visual design, and reviewed the result, while "the tools wrote the code against that specification." That paragraph is not independent evidence. Line-level human versus agent authorship cannot be recovered. Sixteen commits, including the core selection-protocol commit `cdabd95` ("feat: leakage-free selection protocol with baseline gate, real CV, provenance"), carry an explicit arena-agent co-author trailer. Pull requests were opened from `arena/*` branches. Treat personal ownership as: sole human account on the repository, with documented AI co-authorship on a large part of the implementation, and no verifiable split of "decisions I made" versus "code the agent wrote" except where a trailer or the README self-report says so.

---

## Problem

**What problem was actually being solved?**

Predict next-year GDP per capita growth (`NY.GDP.PCAP.KD.ZG`, annual percent) for African country-years from the same country's indicators in the current year, and decide whether a fitted model beats simple baselines on later years.

That problem is implemented, not just described:

- Target construction is `src/features.py:create_target` (grouped shift of the growth column by one year within `iso3`).
- The committed panel `data/processed/model_data.parquet` has 1,300 country-year rows, 52 countries, years 2000–2024, and a `target_next_year` column. 1,205 rows have a non-null next-year target (1,300 × 7.31% missing ≈ 95; 1,300 − 95 = 1,205). Presentation and PDF use 1,205. This review confirmed the missingness rate on the parquet.
- Temporal split is hard-coded in `config/indicators.yaml` (`train_end: 2017`, `val_end: 2020`) and applied by `src/features.py:create_temporal_split`. Committed split sizes in `models/model_metadata.json`: train 905, validation 150, test 150. Target-year windows: train 2001–2018, validation 2019–2021, test 2022–2024.
- Model selection, a baseline gate, one test evaluation, and artifact writing are implemented in `scripts/finalize_model.py`.
- A Streamlit app (`app.py`) loads the saved pipeline and exposes four pages. It does not retrain.

The headline result the repository argues for is a negative one: the deployed model does not beat a historical-average rule by a margin that survives a paired bootstrap. That negative result is computed and stored, and it was reproduced in this review (see ML / AI work).

**Aspirational language that the code does not establish:**

- "Decision-support system" for development analysts, policy teams, NGOs, and government planners (`README.md`, `docs/CAPSTONE_SPEC.md` section 1). No users, deployments-with-traffic, or downstream decisions are recorded. The app is a single-user Streamlit viewer.
- "Which observed development conditions are most informative" (`docs/CAPSTONE_SPEC.md` core question). Permutation importance on the validation split finds 2 of 14 features with a 95% interval above zero, both tiny (`models/feature_importance.parquet`). The app's own Model Performance page says to treat predictions as the mean plus noise.
- Policy or causal use. The README, PDF, and `app.py` (`render_causal_disclaimer`, scenario copy) explicitly reject this. The spec requires that rejection. It is implemented as wording, not as a causal estimator. No causal method exists in the code.
- "54 UN African member states." Config lists 54 member codes plus `ESH` (55 entries; `tests/test_config.py:test_country_list_has_no_duplicates`). The committed panel has 52. Mauritius (`MUS`), Sudan (`SDN`), and Western Sahara (`ESH`) are absent. Eritrea and South Sudan are in the panel but absent from the 150 test predictions because 2021–2023 growth is null for both.
- A tool that "helps predict" growth. The stored result says it does not, beyond the average, in a way distinguishable from chance.

---

## What I personally built

**Confirmed from the repository (account-level, not line-level):**

The `Tessa-Saumu` account owns the repository and is the author or merger of every commit. The tree at `295d665` contains a working data-to-app path:

- Configuration: `config/indicators.yaml`, `src/config.py`
- WDI load/filter/reshape: `src/data.py` (cannot be re-run here; raw file is gitignored and absent)
- Feature/target/split helpers: `src/features.py`
- Models, baselines, metrics, metadata writer: `src/train.py`
- Grouped metrics, bootstrap intervals, permutation importance: `src/evaluate.py`
- Selection protocol: `scripts/finalize_model.py`
- Report-number generator: `scripts/build_report_assets.py`
- PDF renderer: `scripts/build_report_pdf.py` (currently cannot run; markdown source deleted)
- Map geometry builder: `scripts/build_geo_reference.py`
- Streamlit app: `app.py` (1,822 lines, four pages)
- Design system: `src/theme.py`, `src/ui.py`, `src/visualization.py`, `.streamlit/config.toml`, `static/fonts/`
- Tests: 12 modules under `tests/`
- Committed artifacts: `models/*`, `data/processed/*`, `figures/*`, `reports/generated/*`, `reports/capstone_report.pdf`, `reports/presentation_deck.html`, `reports/presentation_deck.pdf`, `assets/*.png`, two notebooks with saved outputs

**Unclear ownership:**

- 16 commits are co-authored by `arena-agent`, including the leakage-control protocol (`cdabd95`), correctness fixes (`0b0d910`), tests around that protocol (`2920aa9`), report-asset generation (`14b90e8`), notebook portability (`cc78b7f`), the scenario-slider bugfix (`2332312`), deployment config (`54d24a9`), and several documentation commits.
- Pull request #3 (`de2034a`, merged 2026-08-29) rebuilt the UI: `src/theme.py`, `src/ui.py`, a rewrite of `src/visualization.py`, app restyle, bundled fonts, Natural Earth-derived geometry, and the related tests. The PR author is `app/arena-ai-coding-agent`; the commit trailer names `arena-agent`. The merge commit is by Theresia Saumu.
- The README self-report assigns decisions and review to the human and implementation to tools. Git cannot confirm that split.
- This review cannot prove that the person reading this file is Theresia Saumu. It can prove only what the git account did.

Do not describe any module as personally written unless the claim is limited to "committed under this GitHub account."

---

## Architecture

**Frontend:**
One Streamlit application, `app.py`. Four in-app pages, not separate routes or a multi-page `pages/` package: Project Overview, Explore Africa, Model Performance, Scenario Explorer (`PAGES` in `app.py`). Navigation is sidebar buttons writing `st.session_state["page"]`. Visual system is "Terracotta Editorial": tokens and CSS in `src/theme.py` (`COLORS`, `build_editorial_css`, `inject_editorial_styles`), HTML/SVG components in `src/ui.py`, charts in `src/visualization.py`, Streamlit chrome tokens in `.streamlit/config.toml`. Fonts are bundled OFL files under `static/fonts/` and referenced as `app/static/fonts/...` with `[server] enableStaticServing = true`. No separate frontend framework. No JavaScript app.

**Backend:**
None as a service. Inference is in-process: `joblib.load("models/growth_model.joblib")` then `pipeline.predict`. Loaders are `@st.cache_resource` / `@st.cache_data` in `app.py` (`load_model`, `load_processed_data`, `load_test_predictions`, `load_feature_importance`, `load_country_metadata`, `load_coordinates`, `load_geometry`). No FastAPI, Flask, or other HTTP API. The spec (`docs/CAPSTONE_SPEC.md` section 15) forbids a separate backend; the code follows that.

**Data:**
World Bank WDI, bulk CSV, not the live API. `src/data.py` expects `data/raw/WDICSV.csv` and `data/raw/WDICountry.csv`. `data/raw/` is gitignored (`.gitignore`). Those files are not in the checkout. Download instructions are in `data/README.md`. What is committed is the already-built panel `data/processed/model_data.parquet` (169,818 bytes, SHA-256 `ffcee5e217e83b325085ee23aef344d0628cf6719525217f99762fd783245a8b`, matching `model_metadata.json`) and `data/processed/country_metadata.csv` (52 countries). Map reference files: `data/reference/africa_geometry.json`, `data/reference/africa_centroids.csv`. `scripts/build_geo_reference.py` can rebuild the geometry from a Natural Earth file; that source file is not committed. Metadata field `wdi_vintage` is the string `"unrecorded (WDI bulk download predating commit)"`.

**ML/AI:**
Classical tabular regression. No deep learning, no LLM calls, no feature store, no training at app runtime. Training entry point is `python scripts/finalize_model.py`, which calls `src.train` and `src.evaluate`. Deployed estimator is a scikit-learn `Pipeline`: `SimpleImputer(strategy="median")` then `HistGradientBoostingRegressor`. Ridge is fit for comparison and for coefficient direction, not deployed. Details under ML / AI work.

**Database:**
None. No SQL, ORM, schema migration, or SQLite file. State is parquet, CSV, JSON, and joblib.

**Infrastructure:**
Intended host is Streamlit Community Cloud, entry point `app.py`, documented in `README.md` Deployment. No Dockerfile, no `docker-compose`, no `.github/workflows`, no `runtime.txt`, no `.python-version`, no `packages.txt`, no secrets file, no GitHub environment, no GitHub deployment objects, no GitHub Pages. `.streamlit/config.toml` sets `headless = true` and disables usage-stats gathering. Python constraint is prose plus `requires-python = ">=3.11"` in `pyproject.toml`. The README says Cloud must be set to 3.11 or 3.12 because numpy 1.26.x has no 3.13 wheel. That pin is not enforced by a file Cloud would read; the README also says Cloud ignores `runtime.txt` and `.python-version`.

**Implemented data and control flow:**

1. Offline, not re-runnable from this checkout: raw WDI CSV → `src.data` filters an explicit ISO3 list, filters indicator codes, melts year columns, checks duplicate `(iso3, year, indicator_code)` keys, pivots to a country-year panel, writes `data/processed/model_data.parquet`.
2. `src.features.create_target` adds `target_next_year`. The committed parquet already contains that column, so `python -m src.features` has been run at least once against this panel.
3. `scripts/finalize_model.py` loads the panel, coverage-filters features on the training-year mask only, splits by feature year, computes validation baselines, runs expanding-window search inside the training period only, picks Ridge vs histogram gradient boosting by validation MAE, refuses to write artifacts if the winner does not beat the validation baselines it computed, refits on training rows only, scores the test split once, writes `models/growth_model.joblib`, `models/model_metadata.json`, `models/test_predictions.parquet`, `models/feature_importance.parquet`, `models/ridge_coefficients.parquet`, `models/cv_results_*.csv`, and refreshes `country_metadata.csv`.
4. `scripts/build_report_assets.py` reads those artifacts and writes `reports/generated/*` and `figures/*.png`.
5. At runtime, `app.py` reads the committed model, metadata, panel, and frozen test predictions. Scenario Explorer builds one row and calls `predict`. No network call is required for a prediction. This review found no `requests` / HTTP client usage in `src/`, `scripts/`, or `app.py`.

---

## ML / AI work

Status labels used below: **verified** means this review recomputed or re-ran it; **documented** means the repo states it and this review did not reproduce it; **absent** means not in the code or artifacts.

**Problem formulation:**
**Verified.** Supervised regression. One row is a country-year. Features are indicators at year `t`. Target is growth at `t+1` for the same country. Not a panel model with country fixed effects, not a forecast of unseen countries, not a causal effect. The spec's selection principle item 4 (`docs/CAPSTONE_SPEC.md` section 11) lists "test-set performance" as a model-selection input. The implementation deliberately does not do that (`scripts/finalize_model.py` module docstring, Step H). The code is stricter than that spec bullet. `tests/test_finalize_model.py:test_selection_uses_validation_not_test` encodes the rule on a synthetic panel and passed in this review.

**Target:**
**Verified.** `NY.GDP.PCAP.KD.ZG`, horizon 1 year (`config/indicators.yaml`, metadata). On the committed panel the current-year growth series has 1,255 non-null values, min −49.13 (Libya 2011), max 91.78 (Libya 2012). Next-year target is non-null on 1,205 rows. Training-target mean on feature years ≤ 2017 is about 1.94. Validation-target mean (feature years 2018–2020, so growth years 2019–2021) is about −0.45, pulled down by feature-year 2019 / growth-year 2020 (mean −4.76). Test-target mean is about 1.87. These means were computed from the parquet in this review. The code comment in `scripts/finalize_model.py` that calls the validation target mean "−4.76" is describing the COVID year, not the whole validation window.

**Features:**
**Verified as used by the model. The coverage-drop story for school enrolment is documented, not reproduced.**

Deployed feature list, 14 names, from `models/model_metadata.json`, in the order the pipeline expects:

`BX.KLT.DINV.WD.GD.ZS`, `EG.ELC.ACCS.ZS`, `FP.CPI.TOTL.ZG`, `FS.AST.PRVT.GD.ZS`, `IT.NET.USER.ZS`, `NE.CON.GOVT.ZS`, `NE.GDI.TOTL.ZS`, `NE.TRD.GNFS.ZS`, `NY.GDP.PCAP.CD`, `NY.GDP.PCAP.KD.ZG` (current-year growth, used as a predictor), `SL.UEM.TOTL.ZS`, `SP.DYN.LE00.IN`, `SP.POP.GROW`, `SP.URB.TOTL.IN.ZS`.

`config/indicators.yaml` lists 14 candidate indicators plus the target. One candidate, `SE.SEC.ENRR` (secondary school enrolment), is not a column in the committed panel. `feature_selection` in metadata says `candidates: 14, kept: 14, dropped: {}`. `reports/generated/table_feature_selection.md` therefore lists those 14 model columns as kept, with `train_coverage_pct` `n/a` (the generator only fills coverage for dropped columns; see `scripts/build_report_assets.py:build_feature_selection_table`).

The PDF and the presentation say `SE.SEC.ENRR` was dropped at 59.9% training coverage, leaving 13 configured indicators plus current-year growth. That sentence is not in `metrics.json` and cannot be recomputed from committed files, because the column is already gone and the metadata drop list is empty. **Documented, not reproduced.**

"14 indicators across 6 themes" (`README.md`, presentation deck) is not supported by the config. The YAML `theme` fields are 13 distinct labels (Economic level, Infrastructure, Technology, Investment, External investment, Openness, Macroeconomic stability, Labour market, Health, Education, Financial development, Public sector, Demographics). The spec table is similar and does not group them into six themes. **Unsupported claim.**

Log1p of GDP per capita is applied only inside the Ridge pipeline (`src/features.py:clip_log1p`, `src/train.py:build_ridge_pipeline`). The deployed histogram-gradient-boosting pipeline does not log-transform. Median imputation is inside the pipeline, so it is refit per `fit` call. No forward-fill. No winsorising. Outlier inspection is described in the spec (section 7) and in notebook `01_data_profiling.ipynb` section 6. Extreme values are retained. No code removes them. That matches the spec's "do not automatically remove outliers" rule. There is no stored test of "are these parsing errors versus real shocks" beyond the notebook's saved narrative.

Tree introspection in this review: the deployed model splits on 8 of 14 inputs (`NY.GDP.PCAP.KD.ZG`, `SP.POP.GROW`, `NY.GDP.PCAP.CD`, `BX.KLT.DINV.WD.GD.ZS`, `IT.NET.USER.ZS`, `SP.DYN.LE00.IN`, `SP.URB.TOTL.IN.ZS`, `FS.AST.PRVT.GD.ZS`). It never splits on electricity, inflation, government consumption, capital formation, trade, or unemployment. `app.py:get_model_responsive_features` and `tests/test_app.py:test_deployed_model_ignores_some_features` encode this. The test passed.

**Baseline:**
**Verified for the numbers. The gate covers two rules, not three.**

Implemented predictors:

1. Global training-mean (`src/train.py:global_mean_baseline`). Test MAE 1.8958.
2. Persistence: next year's growth equals this year's growth (`src/train.py:persistence_baseline`). Test MAE 2.2287. Rows missing current-year growth are dropped before the comparison (`finalize_model.py` Step B / Step H). On the committed test window that drop is Eritrea and South Sudan, all three years, which is why predictions have 50 countries × 3 years = 150, not 52 × 3.
3. Country expanding historical mean up to feature year `t`, no future years (`scripts/finalize_model.py:country_historical_mean_baseline`). Test MAE 1.9424. Zero test rows fell back to the global mean on the re-run in this review.

The automatic gate (`src/train.py:enforce_baseline_gate`, called in Step F) runs on validation and only against the baselines stored in `val_baselines`: global mean and persistence. Country historical mean is scored on the test set only. It is not a ship/no-ship check. The re-run log said `Baseline gate PASSED against all 2 baselines.`

README and the presentation say the build must beat "the trivial rules" and list three rules, on the tuning data, before anything is saved. That overstates the gate. **The two-baseline gate is verified. The three-baseline gate is not what the code does.**

Gate result on the committed artifact, validation MAE: histogram gradient boosting 3.8869 versus global mean 4.0383 (margin 0.151, about 3.7%) and versus persistence 5.0207. `gate.passed` is true. Validation R² of the winner is −0.11. The gate requires a strictly lower MAE, not a meaningful margin and not a positive R².

**Models:**
**Verified.**

- Ridge: median impute, optional log1p on `NY.GDP.PCAP.CD`, `StandardScaler`, `Ridge`. CV-best alpha on the committed run is 3000 (`models/cv_results_ridge.csv`). Validation MAE 4.0043. Not deployed. Coefficients in `models/ridge_coefficients.parquet`. Largest absolute coefficient is FDI at 0.143. All are small. `src/train.py:get_transformed_feature_names` exists so coefficients are not zipped against raw input order after the column transformer moves the log column. A unit test covers that mapping (`tests/test_train.py:test_get_transformed_feature_names_reflects_column_permutation`) and passed.
- HistGradientBoostingRegressor: median impute, then the booster. Deployed hyperparameters, read from the unpickled estimator in this review: `max_depth=2`, `learning_rate=0.03`, `max_iter=200`, `n_iter_=45`, `l2_regularization=1.0`, `early_stopping=True`, `validation_fraction=0.15`, `n_iter_no_change=15`, `random_state=42`. CV table `models/cv_results_hgb.csv` has 12 configurations (2 depths × 3 learning rates × 2 iteration caps). Best mean fold MAE 3.2129, std 0.749, 3 folds. The depth-2 / 0.03 / 100-iteration neighbour is 3.2136, so the win is tiny.
- No other model class. XGBoost is mentioned only as a spec alternative. SHAP is optional in the spec (section 13) and is not imported or used. **Absent, and not claimed by the README.**

Early stopping is explicit because sklearn's `"auto"` stays off below 10,000 rows (`src/train.py:build_hgb_pipeline`; sklearn source in the review environment confirms `do_early_stopping_ = n_samples > 10_000` when `early_stopping == "auto"`). The claim that an earlier version ran 1,000 rounds and overfit is a code comment and README sentence. The earlier artifact is not in the repo. **Documented, not reproduced.** The comment's "86 percent worse than the global mean" (`src/train.py:enforce_baseline_gate`) is likewise not in current artifacts.

Internal early-stopping holdout is a random `train_test_split` of 15% of the rows passed to `fit` (sklearn `_hist_gradient_boosting/gradient_boosting.py`, `random_state=self._random_seed`). It is not a temporal holdout. The saved model is therefore not fit on all 905 training rows; about 15% are used only to decide when to stop. `refit_strategy: "train_only"` means "do not refit on validation," not "every training row is in the boosting fit."

**Validation methodology:**
**Verified for the protocol on the committed panel and on synthetic panels. One advertised number is not what the permanent test checks.**

- Chronological split. No random row split for the official evaluation.
- Expanding-window search inside training years only: `src/train.py:expanding_window_splits`. Config: initial train end 2010, window 2, final train end 2017. Folds are 2000–2010 → 2011–2012, 2000–2012 → 2013–2014, 2000–2014 → 2015–2016. Three folds. Confirmed by `n_folds` in the CV CSVs.
- Winner chosen by validation MAE, not by test MAE and not by R².
- Test split is read in Step H after the gate. Frozen predictions are what the app and notebooks display. Notebook `02_model_evaluation.ipynb` states it does not train; its saved cells load artifacts. This review did not re-execute the notebooks. Saved outputs contain no error cells and quote the same test MAE 1.8217 and the same paired interval.
- Paired bootstrap of per-row absolute-error differences versus the global mean, 5,000 resamples, `numpy` `RandomState(42)` (`finalize_model.py` Step I). This is a row bootstrap, not a country-block or year-block bootstrap. Rows in the same country and nearby years are dependent, so the interval is likely too narrow, not too wide. Even this optimistic interval includes zero.
- Separate percentile bootstrap of test MAE and RMSE, 2,000 resamples, stored in `reports/generated/metrics.json` as `test_bootstrap_cis` (MAE about 1.52 to 2.18). That interval is for the model's own MAE, not for the improvement over the baseline.
- Adversarial test: `tests/test_finalize_model.py:test_selection_uses_validation_not_test` corrupts test-window targets on a **synthetic** panel and checks that model type, validation metrics, gate, CV CSVs, and importance bytes do not change. It passed. The README says corrupting the real test-year outcomes moves test error from 1.82 to 51.40, and attributes that number to this test. The test does not assert 51.40 and does not use the committed panel. **51.40 is documented, not reproduced, and is not a result of the permanent test as written.**

This review re-ran `python scripts/finalize_model.py` against the committed panel into `/tmp/agml-repro` (not the repo). Under Python 3.11.2, scikit-learn 1.9.1, pandas 2.3.3, numpy 1.26.4, these matched the committed metadata exactly: `model_type`, `feature_names`, `metrics`, `gate`, `significance`, `split_sizes`, `split_target_years`, `feature_selection`, `refit_strategy`. Differences were `created_utc` and `git_commit` (now `295d665`, as expected). The original artifact was pickled with scikit-learn 1.9.0. Re-running training reproduced the numbers; unpickling the old file is a separate question, answered below.

**Metrics:**
**Verified** against `models/model_metadata.json`, `reports/generated/metrics.json`, frozen predictions, and the re-run.

Test set, 150 rows, feature years 2021–2023, 50 countries:

| Rule or model | MAE | RMSE | R² | Directional accuracy | Majority rate | Directional skill | Balanced directional accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Global mean | 1.8958 | 2.8379 | −0.0005 | 0.8067 | 0.8067 | 0.000 | 0.500 |
| Persistence | 2.2287 | 4.5209 | −1.539 | 0.7733 | 0.8067 | −0.033 | 0.663 |
| Country historical mean | 1.9424 | 2.8816 | −0.032 | 0.780 | 0.8067 | −0.027 | 0.575 |
| Deployed HGB | 1.8217 | 2.7942 | 0.030 | 0.8067 | 0.8067 | 0.000 | 0.513 |

Paired MAE improvement versus global mean: 0.0740 percentage points. 95% interval [−0.0422, 0.1868]. `significant_at_95: false`. 5,000 resamples.

From `models/test_predictions.parquet`, recomputed in this review: actual mean 1.873, actual std 2.847, actual range −9.63 to 15.15. Predicted mean 1.793, predicted std 0.725, predicted range −1.57 to 4.66. Mean residual (actual − predicted) +0.0804, matching the notebook's saved printout. Predictions are negative on 2 of 150 rows (1.3%). Actual growth is negative on 19.3% of rows. The model is almost a constant-sign, shrunk-toward-the-mean predictor. That is why directional accuracy equals the share of positive years (121/150 = 80.666…%, printed as 80.7% in the README) and why balanced directional accuracy is 51.3%.

Yearly test MAE from `reports/generated/table_yearly_metrics.md`: 2021 feature year 2.06, 2022 1.85, 2023 1.55. Observations per year in the parquet: 50.

Worst absolute errors, from the frozen predictions (also `reports/generated/table_worst_errors.md`): Libya 2021 (13.15 points), Cabo Verde 2021 (12.53), Equatorial Guinea 2022 (11.01), Seychelles 2021 (8.33), Libya 2022 (7.59). These are real rows. The tests that were supposed to keep the report from inventing worst-error countries currently crash because `reports/capstone_report.md` is gone; they did not re-check the PDF.

Permutation importance on the **validation** split, 30 repeats (`finalize_model.py` calls `compute_permutation_importance_with_ci` on `X_val`, not on test): only `NY.GDP.PCAP.CD` (mean 0.046, interval 0.018 to 0.085) and `SP.POP.GROW` (mean 0.017, interval 0.0002 to 0.037) have a lower bound above zero. The other 12 intervals include zero or are exactly zero. Importance is an MAE increase, not a sign. The README states that. Ridge signs are a separate linear fit and are associations, not effects.

Sensitivity, stored and reproduced: refit on train+validation, then score test. Test MAE 2.0341, mean prediction 1.018 versus mean actual 1.873 (about 0.86 points low). The README's "error rises to 2.03 and predictions run 0.86 points low" matches the artifact. That refit is not the deployed model.

**Final result:**
**Verified, and the methodology supports the negative claim the README actually makes. It does not support a success claim.**

The deployed model’s test MAE is 0.07 points lower than predicting the training mean. The paired 95% interval includes a small loss. R² is 0.03. Directional accuracy adds nothing over "always say up." Validation R² is negative. The validation gate passed only because 3.89 is slightly under 4.04, on a window that includes the 2020 crash, against two of three baselines.

Calling this a successful forecasting model would be false. Calling it a careful negative result is what the artifacts support, with the caveats in Limitations (row bootstrap, random early-stopping split, missing raw vintage, two countries missing from the panel, two more missing from the test set, school-enrolment drop not reproducible).

An earlier "we found signal" version is described in the README and in comments labelled C1/C5. It is not stored. Do not cite its metrics as measured results.

---

## Engineering evidence

**Tests:**
12 test modules, 157 test functions, 167 items collected by pytest (parametrized cases included).

This review, 2026-09-25, fresh venv, Python 3.11.2, pytest 9.1.1, dependencies from `requirements.txt` and `requirements-dev.txt` (resolved scikit-learn 1.9.1, Streamlit 1.64.0, pandas 2.3.3, numpy 1.26.4):

`167 collected, 162 passed, 4 failed, 1 xfailed, 21 warnings, 32.67s.`

Failures, all in `tests/test_report_assets.py`, all `FileNotFoundError` for `reports/capstone_report.md`:

- `test_documents_quote_correct_test_mae`
- `test_no_fabricated_worst_error_countries`
- `test_report_never_restores_overturned_claims`
- `test_directional_accuracy_never_alone`

Those tests also expect `presentation/slides_outline.md` (`DOCS` list and the banned-claim list). That file was removed in commit `64bbfb2` (2026-08-29, "docs: moved all docs to reports/"). `reports/capstone_report.md` was removed in `8c99262` (2026-08-29, "docs: finalized documentation"), which added the PDF in its place. The tests were not updated. `scripts/build_report_pdf.py:main` still defaults to the deleted markdown and raises `FileNotFoundError` if it is missing. This review did not run the PDF script; the missing file is sufficient to know the default path fails.

Expected failure: `tests/test_config.py:test_panel_covers_all_configured_countries` is `@pytest.mark.xfail` because the panel predates Mauritius and Sudan. It xfailed, as marked.

What the passing tests actually cover:

- Config: ISO3 list has 55 unique codes, no listed Middle East codes, all 54 UN African member codes present (`tests/test_config.py`).
- Data helpers: ISO3 filter excludes a MENA sample, indicator filter, wide-to-long, numeric cleaning, duplicate warning (`tests/test_data.py`). They use synthetic frames, not the real WDI file.
- Features: shift, coverage filter, train-only coverage, temporal split, drop null targets (`tests/test_features.py`). Synthetic.
- Training: baselines, pipelines, early-stopping flag, expanding windows, hyperparameter search on a toy frame, baseline gate including a regression of the "worse than the mean" case, metadata provenance fields (`tests/test_train.py`).
- Evaluation: grouped metrics, directional accuracy, bootstrap bounds, permutation importance with a noise flag (`tests/test_evaluate.py`).
- `finalize_model.main` on synthetic learnable and noise panels: artifact set, gate pass/fail, non-zero exit code 2 when the gate fails, `--allow-baseline-failure` records the failure, feature contract matches the pipeline, test predictions stay after `val_end`, importance columns exist, selection ignores corrupted test targets, two runs match on metrics (`tests/test_finalize_model.py`). These tests do not use the committed 1,300-row panel.
- App helpers and Streamlit `AppTest`: scenario column order, training-window percentiles, extrapolation warning, inert-slider regression on Ghana, Nigeria, Kenya, South Africa, and Egypt, causal-guardrail text, and render of Overview plus the other three pages (`tests/test_app.py`). Page-render tests passed, with `InconsistentVersionWarning` on unpickle.
- Theme and UI: token list, CSS uses only declared colours, fonts exist on disk, HTML escaping, table formatting (`tests/test_theme.py`, `tests/test_ui.py`).
- Charts: figures return, importance chart mutes non-significant bars, geometry file exists (`tests/test_visualization.py`).
- Path hygiene: no hardcoded absolute paths in shipped Python, no `print()` in production modules, model artifacts not gitignored (`tests/test_hardcoded_paths.py`). Passed.
- Report assets that do not open the missing markdown: `metrics.json` matches `model_metadata.json` for the headline MAE, gate, and significance; generated tables and figure files exist (`tests/test_report_assets.py`, the two tests that passed).

There is no evidence in git of a CI run. Passing status above is this review only.

README says "166 tests." Collection at HEAD is 167, including the xfail. The suite does not pass.

**APIs:**
None. No route table, no OpenAPI spec, no client. The only HTTP surface is whatever Streamlit serves for `app.py`. This review started `streamlit run app.py --server.headless true --server.address 0.0.0.0 --server.port 8501`. `_stcore/health` returned success within about 2 seconds, and `GET /` returned HTTP 200. The process was then stopped. Pages were not clicked in a browser. `AppTest` coverage is the evidence that the four page functions render.

**CI/CD:**
**Absent.** No `.github/` directory. GitHub Actions workflows API: `total_count: 0`. No pre-commit config. No Makefile. Deployment instructions are manual: push to GitHub, connect the repo in Streamlit Community Cloud, set Python 3.11 or 3.12 in the UI.

**Containers:**
**Absent.** No Dockerfile, Compose file, or container registry reference.

**Architecture patterns:**
Present and used, not just named:

- scikit-learn `Pipeline` so imputation is fit with the model (`src/train.py`).
- Temporal split and expanding-window CV separated from the test set (`src/features.py`, `src/train.py`, `scripts/finalize_model.py`).
- Fail-closed artifact write (`SystemExit(2)` if the gate fails, unless `--allow-baseline-failure`).
- Artifact-driven numbers (`scripts/build_report_assets.py`) with tests that were meant to stop prose from drifting. Those tests no longer match the files on disk.
- Streamlit cache around loads.
- Design tokens isolated from page logic (`src/theme.py`, `src/ui.py`). The split is real, but `app.py` is still 1,822 lines and owns page layout, scenario probing, and copy.
- Notebooks are consumers of `src/` and of saved artifacts, not the training entry point. Kernel name recorded in both notebooks: `africa-growth-ml`. `docs/AGENTS.md` documents the editable-install step required to execute them. This review did not re-execute them.

**Data validation:**
Partial. Config is a dataclass loaded from YAML, not a schema library. Tests lock the country list. `src/data.py:check_duplicates` logs conflicting duplicates at WARNING and does not raise. Coverage filter drops columns below 60% non-null on the training mask (`src/features.py:select_features_by_coverage`). Rows with a null target are dropped. The app warns when a scenario slider leaves the training P1–P99 band (`app.py:check_extrapolation_warning`) and clamps slider defaults into the training min/max. There is no pandera/Great Expectations suite, no raw-file checksum in git, and no check that YAML feature codes are columns in the panel. That last gap is why `SE.SEC.ENRR` can be in the config and absent from the parquet without a failing test, except the xfail, which only checks countries.

**Error handling:**
Present in places, not uniform. Missing raw files raise `FileNotFoundError` and the `__main__` blocks exit 1 (`src/data.py`, `src/features.py`). The app calls `st.error` and `st.stop` if the model, metadata, panel, predictions, importance, or country metadata file is missing. The baseline gate refuses to write. Broad `except Exception` in `src/train.py:_git_commit_or_none` is intentional so provenance never breaks a build; a missing git commit becomes `None`. `app.py:get_model_responsive_features` swallows attribute errors and falls back to "all features are responsive," which would reintroduce inert sliders if sklearn's private tree layout changes. Duplicate conflicts are logged, not failed.

**Monitoring:**
**Absent.** `logging` is used in the pipeline and for a few app load messages. No metrics exporter, no error tracker, no health check beyond Streamlit's own, no retraining schedule in code. The presentation's "retrain annually" line is a speaker note, not a job.

**Security:**
Appropriate to a public read-only demo, not a security design. No authentication, no user accounts, no secrets in the tree (`.env` is gitignored; no `secrets.toml`). No live credential use. The model is loaded with `joblib`/`pickle`. That is safe only if the artifact is trusted. It is the project's own committed file. Unpickling across scikit-learn versions warns; this review saw `InconsistentVersionWarning` (pickled 1.9.0, runtime 1.9.1) and then found max absolute difference 0.0 between `pipeline.predict` on the 150 test rows and `models/test_predictions.parquet`. That is evidence the warning did not change these predictions under 1.9.1. It is not a guarantee for other versions. `requirements.txt` allows `scikit-learn>=1.9.0,<2.0`. `pyproject.toml` allows `scikit-learn>=1.3,<2.0` and `streamlit>=1.30`. A reviewer who installs only the package metadata can get a runtime the README itself calls unsafe for unpickling. No CORS lockdown beyond deleting settings the commit message for `54d24a9` says caused a boot warning. No rate limiting. Nothing here demonstrates application security work.

---

## Scale

All figures below were counted from committed files or from GitHub, not from the README's rounding, except where noted.

- Panel: 1,300 rows, 52 countries, 25 years (2000–2024), 52 rows in every year, 0 duplicate `(iso3, year)` keys, 18 columns including `target_next_year`. File size 169,818 bytes.
- Modelling rows with a next-year target: 1,205. Split 905 / 150 / 150.
- Test predictions: 150 rows, 50 countries, feature years 2021, 2022, 2023 (50 each). Missing from test: Eritrea, South Sudan (growth null in those years).
- Config country list: 55 codes. Panel missing `ESH`, `MUS`, `SDN`.
- Model features: 14. YAML candidates not in the panel: 1 (`SE.SEC.ENRR`).
- Hyperparameter evaluations stored: 12 histogram-gradient-boosting configs and 6 Ridge alphas, 3 temporal folds each.
- Deployed models: 1 (`models/growth_model.joblib`, 56,809 bytes). Estimator stopped at 45 of 200 iterations.
- App pages: 4. HTTP API endpoints owned by this repo: 0.
- Tests collected: 167. Test modules: 12. Python under `src/`, `scripts/`, `app.py`, `tests/`: about 9,527 lines (`wc -l`).
- Notebooks: 2, with saved outputs and zero error outputs. Not re-executed in this review. `01_data_profiling.ipynb`: 24 cells (8 code). `02_model_evaluation.ipynb`: 33 cells (17 code).
- Figures: 10 PNGs under `figures/`. Screenshots: 4 PNGs under `assets/`.
- Report PDF: 18 pages, 761,047 bytes. Presentation PDF: 12 pages, 2,065,002 bytes. HTML deck: `reports/presentation_deck.html`.
- GitHub: 44 commits, 4 merged pull requests, 0 issues, 0 releases, 0 Actions workflows, 1 contributor account. History spans 2026-08-26 to 2026-09-01.
- Users, requests, latency, or uptime: **not in the repository.**

---

## Outcome

**What actually worked?**

- A temporal regression protocol that can be re-run from the committed panel and that reproduces the stored metrics, gate, and significance block exactly (this review, `/tmp` re-run).
- A saved model that unpickles and matches its frozen test predictions exactly under scikit-learn 1.9.1, with a version warning.
- A baseline gate that, on synthetic noise, exits 2 and writes nothing (`tests/test_finalize_model.py`, passed).
- A Streamlit app that boots locally (HTTP 200) and whose four pages render under `AppTest`.
- An interface that states the negative result and the causal limit, visible in `assets/project_overview.png` (test MAE 1.82, gain +0.07, interval includes zero) and in `assets/model_performance.png`.
- Scenario controls that are restricted to features that move the prediction for the selected row, with a regression test after an earlier bug where sliders did nothing (`2332312`, `tests/test_app.py:test_scenario_sliders_are_never_inert`, passed).

**What measurable result exists?**

Technical only:

- Test MAE 1.8217 versus global-mean MAE 1.8958. Improvement 0.0740. 95% paired interval includes 0. Not significant at 95%.
- Test R² 0.030. Validation R² −0.112.
- Directional skill versus the majority class: 0.0 on the test set.
- 2 of 14 validation permutation importances distinguishable from zero, both small relative to a validation MAE near 3.9.
- Predictions on the test set have standard deviation 0.73 against an actual standard deviation of 2.85.

**Real-world or product impact:**
None is evidenced. No user counts, no adoption, no policy use, no external evaluation. "Intended users" in the spec and README are a design statement. The live URL exists as a Streamlit Community Cloud app that was asleep when fetched on 2026-09-25. Sleep is not impact, and this review did not wake it or confirm the deployed commit.

---

## Limitations

**What does NOT work?**

- The test suite does not pass. Four document-guard tests error on deleted markdown.
- `python scripts/build_report_pdf.py` cannot rebuild the PDF from the default path. The PDF is a binary snapshot, not a regenerable output, as of `8c99262`.
- The raw WDI ingest cannot be run. `data/raw/` is absent. `wdi_vintage` is unrecorded. A fresh World Bank download is not guaranteed to match the panel hash. `data/README.md` says re-running against a new zip "will produce the full 54-country panel." That is a hope about a file that is not here. `ESH` still has no WDI rows, by the same README.
- The live app was not verified as serving this commit. Fetching `https://africa-growth-ml.streamlit.app/` returned Streamlit's sleep page ("This app has gone to sleep due to inactivity"), not the dashboard.
- The model does not beat the mean by a detectable margin. Shipping it as a forecast would contradict the project's own metadata (`significant_at_95: false`).
- Scenario Explorer will score years the evaluation never covered. `assets/scenario_explorer.png` shows Côte d'Ivoire 2024 indicators and a prediction "for 2025." Feature year 2024 has a null `target_next_year` for every country (no 2025 growth in the panel). The extrapolation warning is about feature ranges, not about leaving the tested year window.
- School enrolment is configured and not in the data. The 59.9% drop figure cannot be checked.
- README documentation links do not match the tree. The documentation section links the label `reports/capstone_report.md` at `reports/capstone_report.pdf`, and the label `presentation/slides_outline.md` at `reports/presentation_deck.pdf`. The PDF targets exist. The markdown paths do not. The layout tree in the README still describes the PDF and HTML deck, so the break is mainly that section plus the test contract.

**What is unfinished?**

- Mauritius and Sudan, acknowledged, with an xfail rather than a data refresh.
- WDI vintage identity.
- CI.
- A runtime pin file that matches the pickle (Python 3.11/3.12, scikit-learn 1.9.0 or a re-pickle under the version you will deploy).
- Regeneration path for the report once the markdown source was deleted.
- Spec items not implemented, and not claimed as done in the README: SHAP, prediction intervals as a first-class app feature (bootstrap exists in the batch pipeline, not as per-country intervals in the app), `models/preprocessing_metadata.json` (the spec's filename; the repo uses `model_metadata.json` instead).
- Notebook re-execution on a clean machine. Outputs are saved. `docs/AGENTS.md` says a named kernel and `pip install -e ".[dev]"` are required. This review did not do that step.

**What would I change?**

Only items forced by evidence already in the tree, not new product ideas:

- Point the document tests and `build_report_pdf.py` at files that exist, or put the markdown back. The current state makes the project's main credibility mechanism fail.
- Make the three-baseline sentence match the two-baseline gate, or add the country-mean rule to the validation gate.
- Remove or re-derive "6 themes," "59.9%," "166 tests," and "51.40" so every public number is either in `reports/generated/` or clearly labelled as not from the artifacts.
- Record the WDI file hash and vintage, or stop saying a fresh download reproduces the study.
- Pin scikit-learn to the pickle version in every install path (`requirements.txt` and `pyproject.toml` currently disagree).
- State in the app when the selected year is outside the evaluated feature years.
- Treat the random 15% early-stopping split as a documented limitation, or pass a temporal validation slice into `fit` if a future retrain is done. Do not silently refit and claim the old test MAE.

**Modelling weaknesses:**

- Negative result on a small, dependent test set (150 country-years, 50 countries, 3 years). Row bootstrap ignores that dependence.
- Test window is an easier regime than validation (target mean 1.87 versus −0.45). Lower test MAE is not evidence the model improved.
- Almost no predictive dispersion (prediction std 0.73 versus actual std 2.85; 2 negative predictions out of 150).
- Median imputation. The README and PDF already say non-reporting is informative. No missingness indicator is modelled.
- No country holdout. Every test country appears in training. The app says so (`assets/model_performance.png`, limitation 02).
- GDP level and population growth dominate a weak fit. Other "development indicators" are unused by the trees or unused within noise.
- Extreme growth values (Libya, Equatorial Guinea, and others) dominate absolute error. No robust loss.
- Early stopping uses a random row split inside training.
- Hyperparameter search is small, which is a virtue here, but the top histogram-gradient-boosting configs are separated by about 0.001 MAE. "Best config" is barely identified. Selection among 12 configs plus 6 alphas on three folds still spends some chance; the README says this.
- Current-year growth is both a feature and the source of the shifted target. The shift is by country, so this is not label leakage in the ordinary sense, but it is a strong persistence feature. Persistence still loses to the mean on the test set, and the model barely uses current-year growth (importance interval includes zero; it is, however, the most-split feature globally). Those two facts can both be true: the trees split on it and the permutation effect on validation is noise.

**Architecture weaknesses:**

- One 1,822-line app module.
- No service boundary, which the spec wanted, so this is not a broken split. It does mean there is no API, auth, or worker story to cite.
- Document generation and the tests that police it have drifted from the published PDF/HTML.
- Two dependency manifests (`pyproject.toml` vs `requirements.txt`) with different floors for scikit-learn, Streamlit, and pyarrow.
- `model_metadata.json` `git_commit` is `2920aa9`, not HEAD `295d665`. Later commits are docs and UI. The re-run showed the numbers do not depend on those later commits. A reviewer who trusts the metadata hash as "this commit" will look at the wrong SHA.
- Private sklearn attribute `_predictors` is part of the slider logic (`app.py:get_model_responsive_features`). It worked under 1.9.1. It is not a public API.

**Reproducibility problems:**

- Raw data and vintage missing.
- Report markdown missing, so the PDF is not regenerable by the committed script.
- Notebooks not re-run here; kernel setup is manual.
- Unpickle version warning.
- Local `git log` in a shallow clone shows one commit. Full history is on GitHub. Anyone judging process from a shallow clone will under-count the work and miss the co-author trailers unless they fetch.

**Broken or dead integrations:**

- No external integration is wired at runtime, so there is no dead API client. The dead paths are internal: `reports/capstone_report.md`, `presentation/slides_outline.md`, `python scripts/build_report_pdf.py` default, and the four tests that read those paths.
- Streamlit Cloud app: URL resolves, app asleep, deployed SHA not recorded in the repo. Commit `54d24a9`'s message says Cloud sign-in was impossible from the environment that added the deployment notes. Later commits and the README claim a live deploy. The sleep page is the only live check in this review.

**Deployment issues:**

- Manual Cloud deploy, no CI gate before it.
- Python version is a console setting, not a file.
- Community Cloud sleeps. A portfolio link can open a wake interstitial. This review did not get past it.
- No health or error monitoring after deploy.

**Technical debt:**

- Comment labels (B2, B7, C1, C5, C6, H1, H2, M3) refer to a review that is not in the tree. Commit `4f82ee9` says a post-remediation review was added; those review files are not at HEAD. The labels are understandable only as history.
- `table_feature_selection.md` records "kept" with coverage `n/a`, which hides the actual training coverage that `table_eda_summary.md` does compute.
- Screenshot `assets/model_performance.png` shows the country widget as "Cote d'Ivoire" and the caption as "ISO3 KEN". `render_sidebar` derives both from the same selectbox index, and the default country is Kenya. This review did not reproduce a runtime bug. The committed screenshot is internally inconsistent. Other screenshots match (Gabon/GAB, Djibouti/DJI, Côte d'Ivoire/CIV).
- Sidebar wraps `HistGradientBoostingRegressor` mid-word in every screenshot. Cosmetic.

**Unsupported claims, if stated without qualification:**

- Tests pass; there are 166 tests and they guard every quoted document number.
- The gate checks all three trivial rules before save.
- Six themes; school enrolment dropped at 59.9% as an artifact-backed fact.
- Corrupting the real test set moves MAE from 1.82 to 51.40, as something `tests/test_finalize_model.py` measures.
- The analysis can be rebuilt from a fresh WDI download to the same countries and the same hash.
- The deployed Cloud app was verified, in this review, to be serving commit `295d665`.
- Any real-world user or policy outcome.
- Sole human implementation of the protocol and the UI.

---

## Current repository status

**Runs locally:**
Yes, for the app and for the training script, with the caveats below. This review did not use a preinstalled environment. It created `/tmp/agml-venv` from `requirements.txt` and `requirements-dev.txt`.

- `streamlit run app.py` bound `0.0.0.0:8501`, health check succeeded, `GET /` returned HTTP 200, then the process was stopped. Browser walkthrough of sliders was not done.
- `scripts/finalize_model.py` completed on the committed panel and matched stored metrics.
- `python -m src.data` was not run. Raw CSV is absent. It will fail with `FileNotFoundError`.
- `scripts/build_report_pdf.py` default path will fail. Not executed; source file missing.
- Notebooks were not re-executed. Saved outputs have no error cells.

**Tests pass:**
No. 162 passed, 4 failed, 1 xfailed, of 167 collected. See Engineering evidence. The failures are missing markdown, not assertion mismatches on model numbers. The two report tests that only read JSON and generated tables passed.

**Deployment:**
Documented as Streamlit Community Cloud at `https://africa-growth-ml.streamlit.app/`. On 2026-09-25 that URL returned the Cloud sleep page, not the app UI. GitHub has no deployment records and no Pages site. No container deploy. No CI deploy. Whether the sleeping app is commit `295d665` is **not determined**.

**README quality:**
High on the result that matters: the negative finding, the split table, the majority-class trap, the causal limit, and the early-stopping footnote match the artifacts. This review checked the README MAE figures against `metrics.json` indirectly: `test_documents_quote_correct_test_mae` reads `README.md` before the missing report, and it failed on the missing file rather than on a README number, so the README's `MAE x.xx` quotes are inside the artifact allow-list.

Weaknesses a reviewer will hit: "166 tests"; documentation links to deleted markdown; "6 themes"; "must beat the trivial rules" when the gate has two; the 51.40 sentence attributed to a test that does not measure it; two install manifests; deployment marked live without a commit SHA. The "Development process" disclosure of AI assistance is a strength, and it should not be dropped.

**Demo available:**
Partial.

- Local app boots.
- Four screenshots in `assets/` show the editorial UI and numbers that match the artifacts (1.82, +0.07, 52 countries, 14 indicators, yearly MAE table). They were added in commit `2bf4e60` (2026-08-29, "docs: added streamlit dash screenshots to assets/"), after the UI refactor. They are credible captures of this app. They are not proof of the current Cloud revision.
- Public URL is a sleeping Streamlit app. Not exercised past the sleep page.
- PDF report (18 pages) and PDF/HTML deck (12 pages) are in `reports/`. The deck names Theresia Saumu and states the negative result first.

---

## Evidence

**Relevant files:**

- `README.md` — public claims, including the negative result and the claims flagged above
- `docs/CAPSTONE_SPEC.md` — assignment-style spec; not all of it is implemented (SHAP optional and absent; test-set performance as a selection criterion intentionally not followed)
- `docs/DASHBOARD_DESIGN.md` — UI spec for the Terracotta system. Tokens are implemented and tested. This review did not diff every spacing rule against the screenshots.
- `docs/AGENTS.md` — contributor rules and notebook kernel setup. Not evidence that those rules were followed on every commit. `tests/test_hardcoded_paths.py` does enforce no `print()` in production modules, and that test passed.
- `config/indicators.yaml` — target, 14 candidates, 55 ISO3 codes, split years, grids
- `src/data.py`, `src/features.py`, `src/train.py`, `src/evaluate.py`, `src/config.py`
- `scripts/finalize_model.py` — selection protocol
- `scripts/build_report_assets.py`, `scripts/build_report_pdf.py`, `scripts/build_geo_reference.py`
- `app.py`, `src/theme.py`, `src/ui.py`, `src/visualization.py`, `.streamlit/config.toml`
- `models/model_metadata.json`, `models/growth_model.joblib`, `models/test_predictions.parquet`, `models/feature_importance.parquet`, `models/ridge_coefficients.parquet`, `models/cv_results_hgb.csv`, `models/cv_results_ridge.csv`
- `data/processed/model_data.parquet`, `data/processed/country_metadata.csv`, `data/README.md`
- `reports/generated/metrics.json` and `reports/generated/table_*.md`
- `reports/capstone_report.pdf`, `reports/presentation_deck.pdf`, `reports/presentation_deck.html`
- `notebooks/01_data_profiling.ipynb`, `notebooks/02_model_evaluation.ipynb`
- `tests/test_*.py` — especially `test_finalize_model.py`, `test_report_assets.py`, `test_app.py`, `test_config.py`
- `assets/project_overview.png`, `assets/explore_africa.png`, `assets/model_performance.png`, `assets/scenario_explorer.png`
- `.gitignore` — ignores `data/raw/`, `*.zip`, does not ignore `models/` or `data/processed/`

**PRs:**

All four are merged. No open pull requests. Bodies are thin except #3.

- #1, merged 2026-08-27, author `Tessa-Saumu`, head `arena/01a04113-africa-growth-ml`, 7 commits, +6,483 / −1,944, 59 files. Title: "Arena/01a04113 africa growth ml". Body: "Review Items all addressed and confirmed."
- #2, merged 2026-08-27, author `Tessa-Saumu`, head `arena/01a040ed-africa-growth-ml`, 15 commits, +9,547 / −2,033, 62 files. Empty body. Includes the leakage-protocol and remediation commits on that branch's history as merged.
- #3, merged 2026-08-29, author `app/arena-ai-coding-agent`, head `arena/01a04633-africa-growth-ml`, 1 commit (`de2034a`), +4,804 / −801, 34 files. Terracotta Editorial UI. Co-authored-by `arena-agent`.
- #4, merged 2026-08-29, author `Tessa-Saumu`, head `arena/01a04caa-africa-growth-ml`, 1 commit, +28 / −32, 5 files. Live-link and screenshot documentation.

Commits after #4, directly on the default branch, include moving docs (`64bbfb2`, `3938ec0`), deleting `reports/capstone_report.md` while adding the PDF (`8c99262`), and the README link edit (`295d665`).

**Issues:**
None. `gh issue list --state all` returned an empty list.

**Screenshots:**

- `assets/project_overview.png` — Overview page, Gabon selected, ISO3 GAB, KPIs 52 / 14 / 1.82 / +0.07 with the interval caveat. Matches metadata.
- `assets/explore_africa.png` — Explore Africa, Djibouti, indicator small multiples. Matches page structure in `render_explore_page`.
- `assets/model_performance.png` — yearly MAE bars and table (2021: 2.058, 2022: 1.852, 2023: 1.555) matching `table_yearly_metrics.md` at display precision. Country widget and ISO3 caption disagree (Côte d'Ivoire vs KEN). Not reproduced as a code bug.
- `assets/scenario_explorer.png` — Côte d'Ivoire 2024, sliders for GDP per capita, population growth, FDI, urban population, domestic credit, note that 8 indicators cannot move the prediction, prediction labelled for 2025. Consistent with responsive-slider logic and with scoring an unevaluated year.

`docs/screenshots/README.md` points at `assets/`. The `docs/screenshots/` directory contains only that README.

**Demo:**

- Local: `streamlit run app.py` started and returned HTTP 200 in this review, then was stopped.
- Public: `https://africa-growth-ml.streamlit.app/` — Streamlit Community Cloud sleep page on 2026-09-25. App content not retrieved.

**Presentation:**

- `reports/presentation_deck.pdf` — 12 pages. Presenter Theresia Saumu, Aug 2026, AnalystLab capstone. States the negative answer first. Also contains the unverified "6 themes" and "59.9%" lines.
- `reports/presentation_deck.html` — same content, source of the PDF claims checked by search (59.9% and "6 themes" are in the HTML).
- `reports/capstone_report.pdf` — 18 pages. States the negative result, the split, the gate narrative, and limitations including Mauritius/Sudan and median imputation. Claims every figure is pasted from `reports/generated/` and that `tests/test_report_assets.py` fails the build if a document quotes an unsupported metric. That test does not read the PDF. It reads markdown files that are not in the tree, and it currently fails.

**Verification performed in this review (not previously recorded in the repo):**

- pytest as cited above.
- Unpickle and predict on all 150 frozen test rows: max absolute difference 0.0, with `InconsistentVersionWarning`.
- `finalize_model.py` re-run to `/tmp/agml-repro`: metrics, gate, significance, splits, and feature-selection block matched. Repo artifacts were not overwritten.
- Panel shape, country set, missingness, and SHA-256 checked against metadata.
- Live URL fetched. Sleep page only.
- GitHub commits, pull requests, issues, workflows, languages, deployments, and Pages queried with `gh`.

---

## Candidate CV bullets

---

## Work required

Concrete remaining tasks only. No new model family, no SHAP, no extra product surface.

### Critical

Problems that make the repository unsafe to cite as finished professional evidence.

- Restore a green test run. Either put `reports/capstone_report.md` and `presentation/slides_outline.md` back, or change `tests/test_report_assets.py` and `scripts/build_report_pdf.py` so they guard the files that are actually published (`reports/capstone_report.pdf` is not parseable by the current test, so the practical fix is a markdown source the generator owns, plus a check that the PDF was built from it). Until that is done, "tests fail if a document disagrees with the model" is false: four tests error, and the PDF/HTML are unchecked.
- Correct the public sentences that are stronger than the code or the artifacts: 166 tests; three trivial rules in the pre-save gate; six themes; school enrolment dropped at 59.9% as if `feature_selection.dropped` contained it; test error 51.40 as a result of `tests/test_finalize_model.py`. These are in `README.md` and/or the deck/PDF.
- Make one install path reproduce the pickle environment. Align `pyproject.toml` with `requirements.txt` (scikit-learn floor 1.9.0, Streamlit floor 1.62, numpy `<2`, Python 3.11 or 3.12). Re-pickle under the version you will deploy, or pin `scikit-learn==1.9.0`, and say which. The current split between manifests is how a reviewer gets a broken or warned unpickle.

### High value

Changes that make the existing evidence harder to dismiss.

- Add a minimal GitHub Actions workflow that runs `pytest` on 3.11 with the runtime pins. The tests already exist. There is no recorded run on GitHub.
- Write the WDI vintage and a hash of the raw file into `model_metadata.json`, or delete the sentence that a fresh download reproduces the panel. Do not claim MUS/SDN will appear until a re-run has produced them.
- If the 59.9% enrolment drop is real, re-ingest so `SE.SEC.ENRR` is in the panel long enough for `finalize_model.py` to record the drop in `feature_selection.dropped`, and regenerate `table_feature_selection.md`. If the raw file is gone, delete the 59.9% sentence.
- Put the country-historical-mean baseline into the validation gate, or narrow the README/deck to the two baselines the gate checks. Do not leave "beat all three before save" in the speaker script.
- Record the deployed commit SHA next to the Streamlit URL after confirming the Cloud app is awake and matches that SHA. The sleep page is not that confirmation.
- Re-execute both notebooks with the documented kernel and commit outputs only if they still match `model_metadata.json`. The saved outputs currently match; they have not been regenerated on this machine.
- Note, in the method section that already discusses early stopping, that the 15% holdout is a random row split. That is a one-paragraph correction, not a new model.

### Optional

- Fix sidebar wrapping of `HistGradientBoostingRegressor` (`app.py` sidebar footer). Visible in all four screenshots.
- If the Côte d'Ivoire / KEN mismatch in `assets/model_performance.png` reproduces, fix the selectbox state and replace the screenshot. If it does not reproduce, replace the screenshot anyway so the committed image is internally consistent.
- When the selected feature year is outside 2021–2023, say that the prediction is outside the evaluated window. The 2024 → 2025 case is already reachable in the UI.
- Fill training coverage on kept features in `build_feature_selection_table` so the generated table is not `n/a`. The numbers already exist in `table_eda_summary.md`.
- Split `app.py` only if you are going to keep editing it. It is not what makes the evidence weak.

---

## Professional Evidence Assessment

1. **What engineering capability does this project prove best?**
Leakage-aware batch ML evaluation, artifact provenance, and a fail-closed training script, plus a small tested UI that loads those artifacts and refuses to over-claim. It does not prove production backend, CI/CD, or platform engineering. The test suite is real and specific (gate refusal, synthetic test-set corruption, inert-slider regression, ISO3 list). It is not currently green, and nothing in GitHub shows it running unattended.

2. **What ML/AI capability does it prove best?**
Evaluation discipline on a temporal tabular problem: baselines, a sealed test, a paired interval, and a reported negative result. It does not prove forecasting skill, causal inference, deep learning, or feature discovery. The strongest modelling judgement in the repo is stopping at "not distinguishable from the mean" instead of tuning until the test set looked good. That judgement is implemented in `scripts/finalize_model.py` and stored in `significance.significant_at_95: false`.

3. **What is the strongest verifiable achievement?**
A reproducible negative result. On 150 held-out country-years the deployed model’s MAE is 1.82 versus 1.90 for the training mean, and the paired 95% interval [−0.04, 0.19] includes zero. This review re-ran the selection script on the committed panel and matched the stored metrics, gate, and interval exactly, and matched all 150 frozen predictions from the pickled pipeline. The app and the report lead with that result rather than hiding it.

4. **What is the biggest credibility weakness?**
The evidence contract is broken at HEAD, and authorship is easy to over-claim. The tests that exist to stop documents drifting from artifacts fail because the markdown they read was deleted while the PDF still says those tests protect it. Beside that, 16 commits including the core protocol are co-authored by an AI agent, the raw World Bank file and vintage are missing, and a handful of public numbers (six themes, 59.9%, 51.40, three-rule gate, 166 passing tests) are not what the code and artifacts show. A careful reader can still verify the negative result. A careless reader, or a broken test run, will not trust the rest.

5. **Is there enough here to justify investing additional time in this project?**
Yes, a small amount, aimed at the evidence contract rather than at a better model. The panel, protocol, tests, app, and negative result are already real. The critical and high-value tasks above are repairs. Spending time searching for a model that "wins" would fight the finding the artifacts support, unless it is framed as a new study with a new sealed test. There is not enough operational surface (no CI, no data refresh path, no verified live revision) to justify a large engineering rewrite.

6. **What type of professional role would this project support as evidence?**
Applied ML or data scientist roles that care about problem formulation, temporal validation, baselines, and shipping a readable demo of the conclusion. It is usable for an internship-to-junior or junior applied-ML portfolio if the AI assistance is disclosed the way the README already discloses it, and if the bullet describes a negative result rather than a forecasting win. It is weak evidence for ML platform, MLOps, backend, or causal-inference roles. It is weak evidence for "I solely implemented a production ML system." It is reasonable evidence for "I can specify, review, and stand behind an evaluation that does not cheat," which is what the README self-report claims and what the artifacts partially support.

7. **What claims should I NOT make publicly based on the current repository?**

- That the model predicts African growth, beats simple rules in a decision-relevant way, or is accurate enough to screen countries for action.
- That 80.7% directional accuracy is skill. It equals the positive-growth base rate. The model predicts negative growth on 2 of 150 test rows.
- That any indicator "drives" growth, or that a slider result is a policy effect.
- That development analysts, NGOs, or governments use it. No usage evidence exists.
- That the study covers all 54 UN African member states, or that the test set has 52 countries. The panel has 52. The test predictions have 50.
- That the full pipeline is reproducible from a World Bank download. The raw file and vintage are not in the repo.
- That all tests pass, or that there are 166 tests guarding every document. Collection is 167; 4 fail; the PDF is not what those tests read.
- That the public Streamlit app was verified as up and equal to the current commit.
- That you solely wrote the implementation. Sixteen commits, including the selection protocol, are co-authored by `arena-agent`, and pull request #3 was opened by an agent app. The honest public claim is the one already in the README: AI-assisted implementation, human-directed, with review. Do not upgrade that to sole authorship.
- That an earlier version's 1,000-round overfit, the "86 percent worse" comment, or the 51.40 corruption figure are measured results stored in this tree.
- That the work includes CI/CD, containers, a database, an API, monitoring, or SHAP.
- That AnalystLab enrolment, dates, or supervision are proven by anything except the project's own PDF, deck, and acknowledgment lines.
