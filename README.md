# Africa Growth Explorer

I wanted to test whether development indicators could predict next year's economic growth across African countries. The useful result was a limit: **this model did not establish a meaningful improvement over simply predicting the historical average** on its fixed test period.

## What I built and contributed

The project uses World Bank World Development Indicators to build a country-by-year table, compare models and explain the results in a four-page Streamlit app. You can explore countries, inspect model errors and adjust inputs to see how the saved model responds.

I chose the question, directed the implementation and visual design, and reviewed the modelling decisions and results with AI-assisted coding tools. The history records AI co-authorship on substantial parts of the implementation. I do not present the code as exclusively hand-written work.

## How I evaluated it

The committed data covers **52 countries, 2000–2024, and 1,300 country-year rows**. Mauritius and Sudan are absent from this data snapshot. Each row uses one year's indicators to predict the following year's GDP-per-capita growth.

Older years train the model; later years tune it; the final test targets are 2022–2024. Missing-value handling is fitted inside each training fold. Settings are selected using temporal cross-validation: train on earlier years, check on the next years, then repeat with more history.

Before saving a model, the validation gate compares it with the global-average and current-growth rules. A third rule, each country's historical average, is reported on the final test. Passing that gate does not guarantee future performance.

## Results

MAE means mean absolute error: the average size of a prediction error, measured here in percentage points of growth.

| Test method | MAE |
|---|---:|
| Gradient-boosted trees | 1.82 |
| Historical global average | 1.90 |
| This year's growth | 2.23 |
| Country's historical average | 1.94 |

The observed model-versus-average improvement was **0.074 percentage points**. Resampling the 150 test rows gave an improvement interval of **−0.0422 to +0.1868**, which crosses zero. That does not establish reliable uplift. The resampling treats country-year rows independently and does not account for shared-country or shared-year effects.

Directional accuracy was **80.7%**, exactly the same as the majority rule of always predicting positive growth. It adds no directional advantage.

On 13 October 2026, rerunning model selection from the committed table matched **all 150 saved predictions and the recorded metrics exactly**. The full suite passed **166 tests**, with one expected failure for missing country coverage. A fresh Windows Python 3.12 installation was checked; scikit-learn is pinned to the saved model's version, 1.9.0.

## Run it

Use Python 3.11 or 3.12. The saved model and app data are committed, so training and downloads are not needed to view the app.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pip install -e ".[dev]"
streamlit run app.py
```

To check or reproduce the saved-data experiment without replacing its artifacts:

```bash
python -m pytest -q
python scripts/finalize_model.py --output-dir work/reproduction
python scripts/build_report_assets.py
```

For raw-data rebuilding, follow [data/README.md](data/README.md), then run `python -m src.data` and `python -m src.features`. A new World Bank download may differ from the original snapshot. Optional notebook execution uses `jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_profiling.ipynb` and the same command for `02_model_evaluation.ipynb`.

## Limits and evidence

The original raw-data download date is missing. The reproducible path starts from the committed table. Internal early stopping uses a random subset even though outer model selection is temporal. Results cover later years for already-seen countries, not unseen countries.

The app's sliders show model sensitivity, not cause and effect. Future scenario years are unevaluated illustrations, not validated forecasts. The [published app URL](https://africa-growth-ml.streamlit.app/) has not been checked for current availability. No policy impact or deployment adoption is claimed.

Read the [report source](reports/capstone_report.md), [presentation outline](presentation/slides_outline.md) and [generated evidence](reports/generated/). PDF/deck exports are historical snapshots. MIT licence: [LICENSE](LICENSE). Thanks to the World Bank and the AnalystLab internship programme for data and project guidance.
