# Africa Growth Explorer

**Can development indicators predict next year's economic growth in Africa? This project tests that carefully, and the answer is no.**

![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)

> **The model is no better than guessing the average.**
>
> Its predictions miss actual growth by **1.82** percentage points on average.
> A rule that ignores every indicator and always predicts the historical
> average misses by **1.90**. That 0.07 lead sounds like a win, but the test
> set holds only 150 country-years — resample it 5,000 times and the margin
> ranges from **0.04 worse** to **0.19 better**. Because that range includes
> losing, the lead cannot be told apart from luck.
>
> So: knowing a country's development indicators this year does not help
> predict its growth next year.
>
> That is the finding, not an apology for one. What this project contributes
> is the machinery that established it — controls that stop future
> information leaking into past decisions, an automatic check that blocks any
> model failing to beat the simple comparisons, and a test set opened exactly
> once. An earlier version of this project, without that machinery, convinced
> itself it had found signal.

---

## What this is

**The question.** Can this year's development indicators — electricity access, inflation, trade, and eleven others — predict next year's GDP per capita growth across African countries?

**What the app does.** Pick a country, see its indicators and the model's growth estimate, then adjust indicators to see how the estimate moves.

**Who it's for.** Development analysts, economic researchers, policy teams, NGOs, students.

> This is a screening and analysis tool, not a basis for final policy decisions. Combine its output with country-specific expertise.

---

## Data

**World Bank World Development Indicators**

- Source: [datatopics.worldbank.org/world-development-indicators](https://datatopics.worldbank.org/world-development-indicators/) — `WDI_CSV.zip`, not committed; download instructions in [`data/README.md`](data/README.md)
- 14 indicators across 6 themes, plus current-year growth used as an input
- **52 of the 54** UN African member states. The configuration lists all 54; Mauritius and Sudan are missing from this data vintage and arrive with the next refresh. The gap is recorded in `data/README.md` and marked as a known failure in the test suite rather than quietly rounded to 54
- 2000–2024, giving 1,300 country-year rows. The exact dataset is fingerprinted so this analysis can be reproduced

**What the model predicts.** GDP per capita growth (annual %) one year ahead. Ghana's 2019 indicators are used to predict Ghana's 2020 growth.

**How the data was divided.** Older years train the model, middle years tune it, and the most recent years test it. Because each row's target is the *following* year's growth, the target years run one year later than the input years — a distinction that matters enough to state explicitly:

| Portion | Input years | Growth years being predicted | Rows |
|---|---|---|---|
| Training | 2000–2017 | 2001–2018 | 905 |
| Tuning | 2018–2020 | 2019–2021 *(includes the COVID crash)* | 150 |
| Testing | 2021–2023 | 2022–2024 *(untouched until the end)* | 150 |

---

## How the model was built

**The final model is a gradient-boosted tree ensemble** — a method that builds many small decision trees, each correcting the errors of the last. It was compared against Ridge regression, a straight-line method, and both were compared against three trivial rules.

The three trivial rules exist to answer one question: is the model actually earning its complexity?

1. **Predict the historical average** every time, ignoring all indicators
2. **Predict this year's growth** as next year's
3. **Predict each country's own past average**

**Tuning without cheating.** Settings were chosen using only the training years, by training on an early stretch and checking against the years immediately after, then widening the stretch and repeating. The folds are 2000–2010 → 2011–12, 2000–2012 → 2013–14, 2000–2014 → 2015–16. The trees were tried at every combination of depth {2, 3} × learning rate {0.01, 0.03, 0.05} × rounds {100, 200}; Ridge at penalty strengths {1 … 3000}. The test years were never involved.

**An automatic stop.** Before any model is saved, it must beat the trivial rules on the tuning data. If it doesn't, the build fails and writes nothing. This is what makes "worse than guessing the average" impossible to ship by accident.

**The test set is opened once.** It is read a single time, at the very end, to produce the final numbers. Nothing about the model is chosen afterwards.

**One detail worth flagging.** scikit-learn's `early_stopping="auto"` quietly switches itself off below 10,000 samples. With 905 training rows, an earlier version ran all 1,000 boosting rounds unchecked and badly overfit. Setting `early_stopping=True` explicitly fixed this — the final model stops at 45 rounds out of a possible 200.

The deployed model, end to end: `SimpleImputer(median) → HistGradientBoostingRegressor(max_depth=2, learning_rate=0.03, max_iter≤200, l2_regularization=1.0, early_stopping=True)`, on 14 indicators, `refit_strategy="train_only"`. Gap-filling sits inside the model rather than before it, so it is refitted on each fold and cannot leak information between them.

### Results

Every number below is generated from the saved model files by `scripts/build_report_assets.py`. None is typed by hand, and `tests/test_report_assets.py` fails the build if any document disagrees with the model.

Lower MAE and RMSE are better; both are average error sizes in percentage points. R² is the share of variation explained, where 0.0 means "no better than always predicting the average".

| Model | Data | MAE | RMSE | R² | Direction right | Always-"up" scores | Gain over always-"up" |
|---|---|---:|---:|---:|---:|---:|---:|
| Predict the average | test | 1.90 | 2.84 | −0.00 | 80.7% | 80.7% | 0.0 pts |
| Predict this year's growth | test | 2.23 | 4.52 | −1.54 | 77.3% | 80.7% | −3.3 pts |
| Predict country's own average | test | 1.94 | 2.88 | −0.03 | 78.0% | 80.7% | −2.7 pts |
| **Gradient-boosted trees (deployed)** | test | **1.82** | **2.79** | **0.03** | 80.7% | 80.7% | **0.0 pts** |
| Gradient-boosted trees | tuning | 3.89 | 5.97 | −0.11 | 52.7% | 51.3% | +1.3 pts |
| Ridge regression | tuning | 4.00 | 6.08 | −0.16 | 51.3% | 51.3% | 0.0 pts |
| Predict the average | tuning | 4.04 | 6.14 | −0.18 | 51.3% | 51.3% | 0.0 pts |

**Is the 0.07-point lead real?** No. Resampling the 150-row test set 5,000 times and re-running the comparison each time puts the margin anywhere between **0.04 worse** and **0.19 better**. A range that includes losing means the lead cannot be separated from chance.

**The 80.7% direction figure is not what it looks like.** The model calls growth up or down correctly 80.7% of the time. But growth was positive in 80.7% of test years, so always saying "up" — using no data at all — scores exactly the same. The gap between the model and that trivial rule is **zero**. Scoring up-years and down-years separately and averaging gives **51.3%**, a coin flip. This is the majority-class trap, and it is why the raw figure never appears alone here.

**Which indicators matter.** Shuffling one indicator's values at random and re-scoring shows how much the model leaned on it. Only **2 of 14** made a reliable difference — GDP per capita and population growth — and both effects are tiny. Scrambling the stronger of the two worsens predictions by **0.046** percentage points, against typical errors near **3.9**: roughly a one percent effect, real enough to measure and far too small to act on. For the other twelve, scrambling the column changed nothing: the model was not using them. This method measures *how much* an indicator matters, never whether its effect is positive or negative, so no "top growth drivers" claims are made from it. Direction comes only from the Ridge model's coefficients, all 0.14 or smaller, and describes association rather than cause.

---

## The application

Four pages, built with Streamlit. It loads the saved model and never retrains or calls the internet.

1. **Overview** — the problem, the data, the model, and the headline result with its caveat stated up front
2. **Explore Africa** — growth trends by country, indicator charts, regional comparison
3. **Model Performance** — how the model scores against the trivial rules, where its predictions land, its largest errors, and which indicators it actually used
4. **Scenario Explorer** — adjust a country's indicators and watch the prediction move. Sliders are bounded by what the model saw in training, and warn when you push past it

---

## Quick start

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"          # notebooks and tests
```

**Run the app** — the trained model is committed, so no training is needed:

```bash
streamlit run app.py             # http://localhost:8501
```

**Rebuild everything from scratch:**

```bash
# 1. Download WDI_CSV.zip into data/raw/ (see data/README.md)
python -m src.data                       # build the country-year table
python -m src.features

# 2. Tune, check against the trivial rules, score once on test, save
python scripts/finalize_model.py

# 3. Regenerate every number quoted in the documents
python scripts/build_report_assets.py

# 4. Tests and notebooks
pytest -q
jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_profiling.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_model_evaluation.ipynb
```

---

## Reproducibility

`random_state=42` throughout. Running `finalize_model.py` twice produces identical numbers — the two `model_metadata.json` files differ only in their timestamp and commit hash.

Each saved model records where it came from, in `models/model_metadata.json`: `created_utc`, `git_commit`, `library_versions`, `panel_sha256` fingerprinting the exact input data, the split sizes and the target years each covers, the `refit_strategy`, and the results of every check it passed.

The notebooks find the project root on their own — an environment variable first, then the nearest `pyproject.toml` — so they run anywhere, and they execute cleanly under `jupyter nbconvert`.

**Proving the test set was never used.** Deliberately corrupt the test-year outcomes and re-run the pipeline. Everything about model selection stays byte-for-byte identical, while the final test error jumps from 1.82 to 51.40. The selection process cannot see the test set — if it could, corrupting the answers would have changed which model won. `tests/test_finalize_model.py` runs this as a permanent check.

---

## Repository layout

```
africa-growth-ml/
├── app.py                      # The Streamlit application
├── config/indicators.yaml      # Indicators, countries, date splits, tuning grid
├── data/
│   ├── README.md               # Download instructions + known gaps
│   └── processed/              # The prepared country-year table
├── figures/                    # All charts
├── models/                     # Trained model, its metadata, frozen predictions
├── notebooks/                  # Data profiling and model evaluation, both executed
├── reports/
│   ├── capstone_report.md      # The full report
│   └── generated/              # Every number quoted in the docs, auto-generated
├── presentation/               # Slide outline and demo script
├── scripts/
│   ├── finalize_model.py       # Tune → check → score once → save
│   ├── build_report_assets.py  # Generates every document number from the model
│   └── build_report_pdf.py     # Converts the report to PDF
├── src/                        # Data loading, features, training, evaluation, charts
└── tests/                      # 86 tests, including guards against the mistakes above
```

---

## Deployment

**Status:** verified locally. The app starts, serves, and its scenario predictions and range warnings have been exercised end to end. A public Streamlit Cloud instance is **not part of this submission**; the repository is ready to deploy and the steps below are the verified path.

1. Push to GitHub
2. Connect at [Streamlit Cloud](https://share.streamlit.io), entry point `app.py`
3. **Set the Python version to 3.11 or 3.12 under "Advanced settings"** before the first deploy. Cloud defaults to 3.12 and ignores `runtime.txt` and `.python-version`, so this is the only place it can be set. The model was built against numpy 1.26.x, which publishes wheels only up to Python 3.12 — on 3.13+ the install fails outright
4. No secrets and no downloads — everything needed at runtime is committed

All four pages run with no backend: the app reads committed files (`models/growth_model.joblib`, the parquet artifacts, `data/processed/`) and calls `pipeline.predict` in-process. There is no database, no API, and no training at runtime. Total artifact footprint is well under a megabyte, comfortably inside Cloud's memory limit.

**Checklist**

- [x] `app.py` at the repository root
- [x] Imports resolve from the repo root; `pip install -e .` exposes `src.*`
- [x] Trained model committed, not excluded by `.gitignore`
- [x] `requirements.txt` installs cleanly, with no test tooling mixed into runtime dependencies
- [x] Runtime pins match the versions the model was pickled with — scikit-learn does not guarantee unpickling across minor versions
- [x] No absolute local paths anywhere, enforced by `tests/test_hardcoded_paths.py`
- [x] No live API calls
- [x] Model and data cached so they load once, not on every interaction
- [x] Streamlit config free of settings that break cloud hosting
- [ ] Screenshots — not captured here; no browser is reachable in this environment. See `docs/screenshots/README.md` for the shot list to add after deploying

---

## Limitations

**This does not show cause and effect.** The model finds patterns linking indicators to later growth. It cannot show that changing an indicator would change growth. When you move a slider, the app answers "what does the model predict for a country described this way?" — not "what would happen if a policy achieved this?". The real world never holds everything else fixed while one number moves. Answering causal questions needs different methods entirely: natural experiments, before-and-after comparisons against a control group, or randomised trials.

**Practical limits**

- **The model ties with the trivial rule, it does not beat it.** Treat any prediction as the historical average plus noise
- **Only tested on new years, not new countries** — every country appears in training
- **COVID sits just before the test window.** The model is trained on pre-2018 data only, decided in advance for this reason. Training on the COVID years instead makes it worse: error rises to 2.03 and predictions run 0.86 points low
- **Small test set** at 150 rows, and 12 configurations were compared, so some of the winning margin is chance
- **52 of 54 countries** in the committed data
- **Gaps are filled with median values**, which discards information — countries that fail to report tend to differ systematically from those that do
- **Out-of-range inputs are flagged, not fixed.** Push a slider beyond what the model saw in training and it warns you, but the prediction is still unreliable

---

## Documentation

- **Full report:** [`reports/capstone_report.md`](reports/capstone_report.md) — run `python scripts/build_report_pdf.py` for a PDF
- **Presentation:** [`presentation/slides_outline.md`](presentation/slides_outline.md)
- **Data sources and known gaps:** [`data/README.md`](data/README.md)

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgments

World Bank for the World Development Indicators; the scikit-learn and Streamlit teams; and the AnalystLab internship programme for project guidance.
