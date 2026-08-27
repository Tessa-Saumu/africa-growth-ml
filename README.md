# Africa Growth Explorer

**A Machine Learning Decision-Support System Using World Bank Development Indicators**

Predicting near-term GDP per capita growth across African countries — and reporting honestly when the data says it can't.

![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)

> **Headline result (verified, not sold):** the deployed model reaches test
> MAE **1.82** against **1.90** for a predict-the-mean baseline. The paired
> 95% CI on that improvement, [−0.04, +0.19], **includes zero** — the 14 WDI
> indicators evaluated here carry no statistically significant information
> about next-year growth beyond the unconditional mean. The project's value is
> the leakage-free protocol that established this (and the gate that makes a
> worse-than-constant model unshippable), demonstrated on a task where an
> earlier cycle *did* manage to fool itself.

---

## Project Overview

### Core Question
> To what extent can recent development indicators predict near-term GDP per capita growth across African countries, and which observed development conditions are most informative for those predictions?

### Decision-Support Question
> Given a country's current development profile, what level of next-year GDP per capita growth does the model estimate, which indicators move that estimate, and how does the estimate change under alternative scenarios?

### Intended Users
Development analysts, economic researchers, policy analysts, government planning teams, NGOs, and students comparing development conditions across African countries.

> **This tool is for screening and analytical support, not final policy decisions.** Combine model output with expert knowledge and country-specific evidence.

---

## Data

**World Bank World Development Indicators (WDI)**
- Source: [datatopics.worldbank.org/world-development-indicators](https://datatopics.worldbank.org/world-development-indicators/) (`WDI_CSV.zip`, not committed — download instructions in [`data/README.md`](data/README.md))
- 14 candidate indicators across 6 themes (plus current-year growth carried as a feature)
- Country list: all **54 UN African member states + Western Sahara (ESH)** in `config/indicators.yaml`; the committed panel currently covers **52** — Mauritius and Sudan arrive on the next WDI re-ingestion (documented in `data/README.md`)
- Time range: 2000–2024; processed panel 1,300 country-year rows, SHA-256 pinned in model metadata

### Target Definition
- **Target:** GDP per capita growth (annual %) in year *t+1* — `NY.GDP.PCAP.KD.ZG`, created by `groupby("iso3").shift(-1)`
- **Splits (feature years → target years):** train 2000–2017 → 2001–2018 · val 2018–2020 → **2019–2021** (includes the COVID target year) · test 2021–2023 → **2022–2024**

---

## Model & Protocol

**HistGradientBoostingRegressor** (selected by expanding-window CV inside the training period, gated against validation baselines, refit on train only under a *pre-registered* policy):

- `SimpleImputer(median) → HGB(max_depth=2, lr=0.03, max_iter≤200, l2=1.0, early_stopping=True)` — early stopping is explicit because sklearn's `"auto"` is inert below 10k samples
- Expanding-window CV grid (spec §9): folds train 2000–2010→val 2011–12, 2000–2012→2013–14, 2000–2014→2015–16; Ridge α over {1…3000}, HGB over depth×lr×iter
- **Baseline gate:** artifacts are only written if the winner beats the global-mean and persistence baselines **on validation** (`enforce_baseline_gate`; failure exits non-zero)
- **Test set is read exactly once** (`scripts/finalize_model.py`, Step H); notebooks load frozen results and never select

### Results (generated from committed artifacts)

`reports/generated/` is produced by `scripts/build_report_assets.py`; no number below is hand-typed, and `tests/test_report_assets.py` fails if documents drift.

| Model | Split | MAE | RMSE | R² | Dir. acc | Majority rate | Dir. skill |
|---|---|---|---|---|---|---|---|
| Global mean baseline | test | 1.90 | 2.84 | -0.00 | 0.81 | 0.81 | 0.00 |
| Persistence baseline | test | 2.23 | 4.52 | -1.54 | 0.77 | 0.81 | -0.03 |
| Country historical mean baseline | test | 1.94 | 2.88 | -0.03 | 0.78 | 0.81 | -0.03 |
| **HGB (deployed)** | test | **1.82** | **2.79** | **0.03** | 0.81 | 0.81 | 0.00 |
| HGB (CV-best) | validation | 3.89 | 5.97 | -0.11 | 0.53 | 0.51 | 0.01 |
| Ridge (CV-best) | validation | 4.00 | 6.08 | -0.16 | 0.51 | 0.51 | 0.00 |
| Global mean baseline | validation | 4.04 | 6.14 | -0.18 | 0.51 | 0.51 | 0.00 |

**Statistical significance:** paired bootstrap over test residuals (5,000 resamples, seed 42): improvement vs the global-mean baseline = **+0.07 pp, 95% CI [−0.04, +0.19] — not significant.** The model is statistically indistinguishable from predicting the unconditional mean.

**Directional accuracy requires the majority-class rate:** 80.7% of test targets are non-negative, so *any* always-positive predictor scores 80.7%. The deployed model's directional skill is 0.00 pp and balanced directional accuracy is 51.3% — no sign information beyond the class prior.

### What Drives Predictions (honest attribution)
Permutation importance on validation, with 95% CIs: **2 of 14 features are distinguishable from zero** — GDP per capita (0.046 [0.018, 0.085]) and population growth (0.017 [0.000, 0.037]); the other 12 straddle zero and are reported as noise. Permutation importance has **no directional meaning** — no "positive/negative driver" claims are made from it. Direction comes only from standardized Ridge coefficients (association, not causation); all |coefficients| are ≤ 0.14 at the CV-selected α=3000, itself a sign of weak linear signal.

---

## Streamlit Application

### 4 Pages
1. **Project Overview** — problem, data, model, honest headline metrics, causal disclaimer
2. **Explore Africa** — country trends, indicator charts, regional comparison table
3. **Model Performance** — significance banner, baseline comparison (test + validation), actual-vs-predicted, residuals, CI-gated feature importance with noise table, Ridge-direction panel, by-year metrics
4. **Scenario Explorer** — interactive what-if analysis, **training-window** extrapolation guardrails, one-at-a-time model-delta table (no fake "importance × change" arithmetic)

The app loads serialized artifacts only (pipeline, frozen predictions, importance) — it never retrains and makes no network calls.

---

## Quick Start

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"          # dev extras: notebooks/tests
```

### Run the App (model artifacts are committed — no training required)
```bash
streamlit run app.py             # http://localhost:8501
```

### Reproduce the Full Pipeline
```bash
# 1. Download WDI_CSV.zip from the World Bank into data/raw/ (see data/README.md)
# 2. Build the country-year panel
python -m src.data
python -m src.features

# 3. Select, gate, and finalize the model (writes models/*)
python scripts/finalize_model.py

# 4. Regenerate report assets + the PDF report from the artifacts
python scripts/build_report_assets.py
python scripts/build_report_pdf.py       # reports/capstone_report.pdf

# 5. Run the test suite / re-execute notebooks (dev extras)
pytest -q
jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_profiling.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_model_evaluation.ipynb
```

### Run Tests
```bash
pytest tests/ -q                 # runtime deps: requirements.txt; test deps: requirements-dev.txt
```

---

## Reproducibility

- `random_state=42` everywhere; the finalize script is bit-level deterministic (verified: double-run `model_metadata.json` comparison of winner-test metrics)
- Provenance in `models/model_metadata.json`: `created_utc`, `git_commit`, `library_versions`, `panel_sha256`, split sizes, **split target-year windows** (feature-vs-target-year ambiguity is spelled out), `refit_strategy`, gate + significance + sensitivity blocks, feature-selection audit trail
- Notebooks resolve the project portably (`PROJECT_ROOT` env var → nearest `pyproject.toml`) and execute cleanly via `jupyter nbconvert` against the registered `africa-growth-ml` kernel

---

## Project Structure

```
africa-growth-ml/
├── app.py                      # Streamlit application (entry point)
├── config/indicators.yaml      # Indicators, countries, splits, CV grid
├── data/
│   ├── README.md               # Download instructions + known data limitations
│   └── processed/              # Committed panel + country metadata
├── models/                     # Committed artifacts: pipeline, metadata,
│   ├── growth_model.joblib     #   frozen test predictions, importance,
│   ├── model_metadata.json     #   coefficients, CV results
│   └── ...
├── notebooks/                  # Executed, artifact-driven (no logic)
├── figures/                  # all PNG figures (notebooks + report assets)
├── reports/
│   ├── capstone_report.md      # Final report (numbers pasted from generated/)
│   └── generated/              # metrics.json + tables (generated)
├── scripts/
│   ├── finalize_model.py       # Selection → gate → sealed test → artifacts
│   └── build_report_assets.py  # All document numbers, from artifacts
├── src/                        # config, data, features, train, evaluate, visualization
└── tests/                      # 60+ tests incl. finalize/app regression guards
```

---

## Deployment

**Status:** verified locally (app boots headless, HTTP 200, scenario prediction and extrapolation warnings exercised); a public Streamlit Cloud instance is **not included in this submission** — the repository is deployment-ready (entry point `app.py`, committed artifacts, `requirements.txt`, Python 3.11) and the steps below are the verified path.

1. Push this repository to GitHub
2. Connect it to [Streamlit Cloud](https://share.streamlit.io), entry point `app.py`, Python 3.11
3. No secrets or data downloads needed — all runtime artifacts are committed

### Deployment Checklist
- [x] `app.py` at repository root
- [x] All imports resolve from repo root; package installable (`pip install -e .` exposes `src.*`)
- [x] Model files committed (not gitignored)
- [x] `requirements.txt` installs cleanly (no test tooling in runtime deps)
- [x] No local absolute paths (grep-verified incl. notebooks; enforced by `tests/test_hardcoded_paths.py`)
- [x] No live API calls
- [x] `@st.cache_resource` for the model, `@st.cache_data` for data
- [x] `.streamlit/config.toml` free of `enableCORS`/`port` overrides
- [ ] App screenshots — **not captured in this environment** (no browser
      binary reachable); see `docs/screenshots/README.md` for why and for the
      exact capture list to add after deployment

---

## Important Limitations

### Causal Interpretation Disclaimer
> This application uses machine learning for **prediction and decision support**, not causal policy-effect estimation. The model identifies statistical associations between development indicators and future GDP per capita growth. It **cannot prove** that changing an indicator causes a change in growth. Scenario deltas are *conditional predictions*, not counterfactuals: the real world does not hold other factors constant when one slider moves. Establishing policy effects requires experimental or quasi-experimental designs (IV, DiD, RCTs, synthetic controls, explicit causal models).

### Technical Limitations
- **Parity, not victory:** the headline paired CI spans zero — treat point predictions as the mean plus noise
- **Temporal generalization only:** future years for seen countries, not unseen countries
- **COVID regime break:** validation target years include the 2020 crash; refit policy is pre-registered (`train_only`) and the train+val counterfactual is reported as sensitivity
- **Small test set:** n=150; multiple-comparison exposure from the CV grid is discussed in report §13
- **52-of-54 country coverage** in the committed panel (MUS/SDN pending WDI re-ingestion)
- **Median imputation** ignores informative missingness
- **Extrapolation risk:** scenario guardrails warn outside the *training* P1–P99 band, but no mechanism repairs unsupported regions

---

## License

MIT License — see [LICENSE](LICENSE).

## Acknowledgments

- World Bank for the World Development Indicators
- scikit-learn team for HistGradientBoostingRegressor
- Streamlit team for the deployment platform
- FlyRank internship program for project guidance
