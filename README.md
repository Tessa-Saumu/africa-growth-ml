# Africa Growth Explorer

**A Machine Learning Decision-Support System Using World Bank Development Indicators**

Predicting near-term GDP per capita growth across African countries to support development analysis and policy screening.

![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)

---

## Project Overview

### Core Question
> To what extent can recent development indicators predict near-term GDP per capita growth across African countries, and which observed development conditions are most informative for those predictions?

### Decision-Support Question
> Given a country's current development profile, what level of next-year GDP per capita growth does the model estimate, which indicators contribute most to that estimate, and how does the estimate change under alternative development scenarios?

### Intended Users
- Development analysts
- Economic researchers
- Policy analysts
- Government planning teams
- NGOs and development institutions
- Students and researchers comparing development conditions across African countries

> **This tool is designed for screening and analytical support. It is not intended to make final policy decisions.** Always combine model output with expert knowledge and country-specific evidence.

---

## Data Source

**World Bank World Development Indicators (WDI)**
- Source: [datatopics.worldbank.org/world-development-indicators](https://datatopics.worldbank.org/world-development-indicators/)
- Raw data: `WDI_CSV.zip` (not committed; download from World Bank)
- 14 candidate indicators across 6 themes
- 52 African countries (UN-recognized states)
- Time range: 2000-2024

### Target Definition
- **Target:** GDP per capita growth (annual %) in year *t+1*
- **WDI Code:** `NY.GDP.PCAP.KD.ZG`
- Created by shifting country-year growth forward by 1 year within each country

### Features Used (14)
| Code | Name | Theme |
|------|------|-------|
| `EG.ELC.ACCS.ZS` | Access to electricity (% of population) | Infrastructure |
| `IT.NET.USER.ZS` | Individuals using the Internet (% of population) | Technology |
| `NE.GDI.TOTL.ZS` | Gross capital formation (% of GDP) | Investment |
| `BX.KLT.DINV.WD.GD.ZS` | Foreign direct investment, net inflows (% of GDP) | External investment |
| `NE.TRD.GNFS.ZS` | Trade (% of GDP) | Openness |
| `FP.CPI.TOTL.ZG` | Inflation, consumer prices (annual %) | Macroeconomic stability |
| `SL.UEM.TOTL.ZS` | Unemployment, total (% of total labor force) | Labour market |
| `SP.DYN.LE00.IN` | Life expectancy at birth, total (years) | Health |
| `FS.AST.PRVT.GD.ZS` | Domestic credit to private sector (% of GDP) | Financial development |
| `NE.CON.GOVT.ZS` | General government final consumption expenditure (% of GDP) | Public sector |
| `SP.URB.TOTL.IN.ZS` | Urban population (% of total population) | Demographics |
| `SP.POP.GROW` | Population growth (annual %) | Demographics |
| `NY.GDP.PCAP.CD` | GDP per capita (current US$) | Economic level |
| `NY.GDP.PCAP.KD.ZG` | GDP per capita growth (annual %) | Economic performance |

Features selected by ≥60% coverage on training data (2000-2017). `NY.GDP.PCAP.CD` is log-transformed inside the pipeline.

---

## Model

### Algorithm
**HistGradientBoostingRegressor** (scikit-learn)
- `max_iter=1000`, `learning_rate=0.05`, `max_depth=5`, `random_state=42`

### Preprocessing
- Median imputation (fitted on training data only)
- Log1p transform for GDP per capita (applied inside pipeline via `FunctionTransformer`)
- No scaling needed for tree-based model

### Validation Strategy
**Temporal split (no random splitting to avoid leakage):**
- Training: 2000-2017
- Validation: 2018-2020 (includes 2020 COVID shock)
- Test: 2021+ (post-COVID)

### Baselines
1. **Global Mean:** Predict average training growth for all test observations
2. **Persistence:** Predict next year's growth = current year's observed growth

---

## Model Performance (Test Set)

| Model | MAE (pp) | RMSE (pp) | R² | Directional Accuracy |
|-------|----------|-----------|-----|---------------------|
| Global Mean Baseline | 1.90 | 2.84 | -0.00 | 80.7% |
| Persistence Baseline | 2.23 | 4.52 | -1.54 | 77.3% |
| **HistGradientBoosting (Test)** | **3.54** | **5.00** | **-2.10** | **52.7%** |

> **Note:** Negative R² indicates the model does not outperform a horizontal line (global mean) in terms of explained variance. This is common for noisy macroeconomic prediction tasks. The model's value is in directional screening and scenario exploration, not point prediction accuracy.

### Feature Importance (Permutation, Test Set)
1. **Electricity Access** (+0.60) - strongest positive predictor
2. **GDP per Capita (log)** (+0.22)
3. **Unemployment** (+0.18)
4. **Domestic Credit to Private Sector** (+0.06)
5. **Gross Capital Formation** (-0.21)
6. **Inflation** (-0.20)
7. **Population Growth** (-0.15)
8. **Life Expectancy** (-0.15)
9. **Internet Usage** (-0.14)
10. **Urban Population** (-0.09)
11. **GDP Growth (current year)** (-0.09)
12. **Trade Openness** (-0.08)
13. **FDI Inflows** (-0.08)
14. **Government Consumption** (-0.06)

*Positive importance = higher values associated with higher predicted growth; negative = inverse association.*

---

## Streamlit Application

### 4 Pages
1. **Project Overview** - Problem statement, data, model, metrics, causal disclaimer
2. **Explore Africa** - Country trends, indicator charts, regional comparison table
3. **Model Performance** - Baseline comparison, actual vs predicted, residuals, feature importance, yearly metrics
4. **Scenario Explorer** - Interactive what-if analysis with extrapolation warnings

### Key Features
- **B3 FIX:** Feature list read from `model_metadata.json` (single source of truth)
- **B4 FIX:** Slider ranges = full observed data min/max; warnings fire at P1-P99 boundaries
- **B5 FIX:** Precomputed test predictions & permutation importance loaded from parquet (not recomputed per rerun)
- Causal disclaimer on Overview and Scenario pages
- Guards against IndexError when country has no growth data
- Uses project visual language from `src.visualization`

---

## Quick Start

### Prerequisites
- Python 3.11+
- pip

### Installation
```bash
git clone https://github.com/yourusername/africa-growth-ml.git
cd africa-growth-ml
pip install -r requirements.txt
```

### Run the App
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

### Run Tests
```bash
pytest tests/ -v
```

### Reproduce Full Pipeline
```bash
# 1. Download WDI_CSV.zip from World Bank and extract to data/raw/
# 2. Run data processing
python -m src.data

# 3. Run feature engineering
python -m src.features

# 4. Train model
python -m src.train

# 5. Evaluate
python -m src.evaluate

# 6. Launch app
streamlit run app.py
```

---

## Project Structure

```
africa-growth-ml/
├── app.py                      # Streamlit application (entry point)
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── pyproject.toml              # Package metadata
├── .gitignore
├── LICENSE
├── config/
│   └── indicators.yaml         # Indicator definitions, countries, temporal splits
├── data/
│   ├── README.md               # Data download instructions
│   ├── raw/                    # Raw WDI files (not committed)
│   └── processed/
│       ├── model_data.parquet  # Country-year panel (1300 rows, 52 countries)
│       └── country_metadata.csv
├── models/
│   ├── growth_model.joblib     # Fitted sklearn pipeline
│   ├── model_metadata.json     # Feature contract + metrics (single source of truth)
│   ├── test_predictions.parquet # Precomputed test predictions
│   └── feature_importance.parquet # Precomputed permutation importance
├── notebooks/
│   ├── 01_data_profiling.ipynb # EDA notebook
│   └── 02_model_evaluation.ipynb # Model evaluation notebook
├── reports/
│   └── capstone_report.md      # Final report (markdown)
├── presentation/
│   └── slides_outline.md       # Presentation outline
├── src/
│   ├── __init__.py
│   ├── config.py               # Configuration loading
│   ├── data.py                 # Data loading, filtering, reshaping
│   ├── features.py             # Feature engineering, target creation, temporal splits
│   ├── train.py                # Model training, pipelines, serialization
│   ├── evaluate.py             # Evaluation, error analysis, feature importance
│   └── visualization.py        # Reusable charts with project visual language
└── tests/
    ├── test_config.py
    ├── test_data.py
    ├── test_features.py
    ├── test_train.py
    ├── test_evaluate.py
    └── test_visualization.py
```

---

## Deployment

### Streamlit Cloud
1. Push repository to GitHub
2. Connect to [Streamlit Cloud](https://streamlit.io/cloud)
3. Deploy from `main` branch with `app.py` as entry point
4. No additional configuration needed (model files committed)

### Deployment Checklist
- [x] `app.py` at repository root
- [x] All imports work from repository root
- [x] Model files committed (not gitignored)
- [x] `requirements.txt` installs successfully
- [x] No local absolute paths
- [x] No live API calls
- [x] Python 3.11 compatible
- [x] Uses `@st.cache_resource` for model, `@st.cache_data` for data

---

## Important Limitations

### Causal Interpretation Disclaimer
> This application uses machine learning for **prediction and decision support**, not causal policy-effect estimation.
>
> The model identifies statistical associations between development indicators and future GDP per capita growth. It **cannot prove** that changing one indicator will cause a particular change in growth.
>
> For example: If increasing electricity access from 70% to 80% in the Scenario Explorer leads to a higher predicted growth, this means the model *associates* that feature profile with higher predicted growth. It does **not** prove that increasing electricity access alone will cause the predicted increase.
>
> This distinction between **prediction**, **association**, **causality**, and **intervention effects** is fundamental to responsible use of this tool.

### Technical Limitations
- **Temporal generalization only:** Evaluates prediction for future years of countries seen during training, not for completely unseen countries
- **Negative R² on test set:** Model does not outperform global mean in explained variance
- **Association ≠ Causation:** Feature importance reflects predictive association, not causal effect
- **COVID-19 period:** Validation (2018-2020) includes 2020 shock; test (2021+) is post-COVID
- **Limited features:** Only 14 WDI indicators; many growth determinants omitted
- **Median imputation:** Missing values filled with training median, not country-specific
- **Extrapolation risk:** Scenario predictions outside historical P1-P99 ranges are flagged but may be unreliable

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- World Bank for World Development Indicators data
- scikit-learn team for HistGradientBoostingRegressor
- Streamlit team for the deployment platform
- FlyRank internship program for project guidance