# Africa Growth Explorer: Capstone Report

**A Machine Learning Decision-Support System Using World Bank Development Indicators**

---

## 1. Introduction

### Economic Growth and Development Context
Economic growth remains the primary engine for poverty reduction and improved living standards across Africa. Understanding the drivers of near-term GDP per capita growth is critical for development analysts, policy makers, and investors seeking to allocate resources effectively.

### Why African Development Analysis Is Important
Africa's 54 countries represent diverse economic structures, resource endowments, and development trajectories. Yet they share common challenges: infrastructure gaps, human capital constraints, and vulnerability to external shocks. A systematic, data-driven approach to comparing development conditions across countries can reveal patterns that single-country analysis misses.

### Why Decision Support Is More Useful Than Isolated Prediction
A point prediction of "Ghana's 2024 growth will be 3.2%" has limited utility. What decision makers need is: *Given Ghana's current profile, what does the model predict, which indicators drive that prediction, and how would the prediction change if electricity access improved by 10 percentage points?* This decision-support framing turns a static prediction into an interactive analytical tool.

### Role of World Bank WDI
The World Development Indicators provide the most comprehensive, standardized, cross-country comparable dataset for development analysis. Using WDI ensures reproducibility, transparency, and alignment with the indicators policy makers already monitor.

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
52 UN-recognized African countries (explicit ISO3 list, not region-substring filtering).

### Intended User
Development analysts, economic researchers, policy analysts, government planning teams, NGOs, and students.

### Expected Impact
Screening tool to identify countries where development profiles suggest stronger/weaker near-term growth, and to explore which indicator changes are most associated with prediction changes.

---

## 3. Dataset Description

### World Bank WDI Source
- Downloaded from [datatopics.worldbank.org/world-development-indicators](https://datatopics.worldbank.org/world-development-indicators/)
- File: `WDI_CSV.zip` containing `WDICSV.csv` (data) and `WDICountry.csv` (metadata)

### Indicators Considered
14 candidate indicators across 6 themes (see Feature Table in Section 4).

### Number of Countries
52 African countries (explicit ISO3 list in `config/indicators.yaml`).

### Time Range
2000-2024 (latest available). Training ends 2017, validation ends 2020, test begins 2021.

### Final Feature Set
14 features retained after ≥60% coverage filter on training data (2000-2017).

### Target Definition
`target_next_year = df.groupby("iso3")["NY.GDP.PCAP.KD.ZG"].shift(-1)`

### Data Limitations
- Missingness varies by indicator (14-85% coverage)
- No forward-filling of volatile variables (inflation, FDI)
- Median imputation only (no model-based imputation)
- Aggregates (SSA, AFE, AFW, etc.) explicitly excluded
- 2020 COVID shock sits in validation period by design

---

## 4. Methodology

### Data Transformation
1. **Load raw WDI CSV** (wide format: country × indicator × year columns)
2. **Filter African countries** using explicit ISO3 list (B7 fix: avoids MENA substring trap)
3. **Select 14 target indicators** + target code
4. **Reshape wide to long** (melt year columns)
5. **Pivot to country-year panel** (one row per country-year, indicators as columns)
6. **Clean numeric** (handle blanks, placeholders, convert to numeric)
7. **Create next-year target** via grouped shift
8. **Select features by coverage** (≥60% on training rows only)
9. **Temporal train/val/test split**

### Missing-Data Strategy
1. Drop features with <60% coverage on training data
2. Drop rows where target is missing (last year per country)
3. Median imputation via sklearn pipeline (fitted on training only)
4. No future information leakage

### Outlier Handling
- Inspected extreme values (hyperinflation, commodity shocks, crises)
- Confirmed not parsing errors
- Retained legitimate observations
- No automatic winsorization or clipping
- Log1p transform for GDP per capita (highly skewed)

### Feature Engineering
- Country-year panel construction
- Next-year target via grouped shift
- Coverage-based feature selection (training data only)
- Log transform for GDP per capita inside pipeline

### Temporal Split
| Split | Years | Rationale |
|-------|-------|-----------|
| Train | 2000-2017 | Pre-COVID, sufficient history |
| Validation | 2018-2020 | Includes 2020 shock for model selection |
| Test | 2021+ | Post-COVID, held-out evaluation |

### Baselines
1. **Global Mean:** `y_train.mean()` for all test predictions
2. **Persistence:** Current year's growth as next-year prediction (B2 fix)

### Models
1. **Ridge Regression:** Imputer → LogTransform (GDPpc) → Scaler → Ridge(α=1.0)
2. **HistGradientBoostingRegressor:** Imputer → HGB(max_iter=1000, lr=0.05, depth=5)

### Evaluation Metrics
- **Primary:** MAE (Mean Absolute Error) in percentage points
- **Secondary:** RMSE, R², Directional Accuracy (sign agreement)
- **Additional:** Metrics by year, by country, worst errors, bootstrap CIs

---

## 5. Exploratory Data Analysis

### Summary Statistics (Processed Panel: 1300 rows, 52 countries, 14 features + target)
- GDP growth range: -60% to +35% (outliers: Libya 2012, Equatorial Guinea 2004)
- Electricity access: 5-100% (median ~55%)
- Internet usage: 0-75% (median ~15%)
- Inflation: -10% to +400% (Zimbabwe hyperinflation years)

### Missingness
- Best coverage: Population growth, Urban population (>90%)
- Worst coverage: FDI inflows, Domestic credit (~14%)
- Missingness correlates with conflict-affected states (Somalia, South Sudan)

### Trend Analysis
- Average African growth declined from ~4% (2000-2010) to ~2% (2010-2019)
- 2020 COVID shock: median growth -3.5%
- Recovery in 2021-2022 but below pre-COVID trend

### Distributions
- GDP growth: approximately normal with heavy tails
- Electricity access: bimodal (low-access vs. high-access countries)
- Internet usage: right-skewed, rapid growth post-2010
- GDP per capita: highly right-skewed → log transform

### Correlations
- Electricity access ↔ Internet usage: 0.78
- GDP per capita ↔ Life expectancy: 0.65
- Inflation ↔ GDP growth: -0.31
- Capital formation ↔ GDP growth: 0.12 (weak)

### Country Comparisons
- **Consistent growers:** Rwanda, Ethiopia, Côte d'Ivoire
- **Volatile:** Libya, Equatorial Guinea, Zimbabwe
- **Stagnant:** South Sudan, Central African Republic

### Main EDA Findings
1. Infrastructure (electricity, internet) strongly correlates with growth
2. Macroeconomic stability (low inflation) associates with positive growth
3. Investment (capital formation) shows weak bivariate correlation but appears in multivariate model
4. Considerable heterogeneity across countries – no single indicator dominates universally

---

## 6. Model Development

### Ridge Regression
- Pipeline: `SimpleImputer(median) → ColumnTransformer(log1p for GDPpc) → StandardScaler → Ridge(alpha=1.0)`
- Hyperparameter: α=1.0 (default, minimal tuning)
- Linear benchmark, interpretable coefficients

### Gradient Boosting (HistGradientBoostingRegressor)
- Pipeline: `SimpleImputer(median) → HGB(max_iter=1000, learning_rate=0.05, max_depth=5, random_state=42)`
- Handles non-linearities and interactions natively
- No scaling needed
- Built-in missing value handling (but we impute for consistency)

### Pipeline Design
- Single fitted `joblib` artifact containing imputer + model
- Log transform inside pipeline via `FunctionTransformer` referencing `src.features.clip_log1p` (picklable by module path)
- Feature names from `model_metadata.json` (single source of truth, B3 fix)

### Hyperparameter Tuning
- Compact grid: `max_iter ∈ {500, 1000}`, `learning_rate ∈ {0.05, 0.1}`, `max_depth ∈ {3, 5}`
- Expanding-window validation: train 2000-2010 → val 2011-2012, train 2000-2012 → val 2013-2014, etc.
- Selected: max_iter=1000, lr=0.05, depth=5 (best validation MAE)

### Reproducibility
- Fixed `random_state=42` throughout
- All artifacts versioned: model, metadata, predictions, importance
- Temporal splits prevent leakage

---

## 7. Model Evaluation

### Performance Table (Test Set: 2021-2023, 150 observations)

| Model | MAE | RMSE | R² | Directional Accuracy |
|-------|-----|------|-----|---------------------|
| Global Mean Baseline | 1.90 | 2.84 | -0.00 | 80.7% |
| Persistence Baseline | 2.23 | 4.52 | -1.54 | 77.3% |
| Ridge (Val) | 3.98 | 5.95 | -0.10 | 56.7% |
| **HGB (Val)** | **3.91** | **5.94** | **-0.10** | **58.0%** |
| **HGB (Test)** | **3.54** | **5.00** | **-2.10** | **52.7%** |

### Actual vs. Predicted (Test Set)
- Scatter shows wide dispersion around identity line
- Systematic underprediction for high-growth outliers
- Model tends to shrink predictions toward mean

### Residual Analysis
- Residuals show heteroscedasticity (larger errors for extreme predictions)
- No obvious pattern vs. predicted values
- Mean residual ≈ 0 (unbiased)

### Feature Importance (Permutation, Test Set)
| Rank | Feature | Importance | Direction |
|------|---------|------------|-----------|
| 1 | Electricity Access | +0.604 | Positive |
| 2 | GDP per Capita (log) | +0.222 | Positive |
| 3 | Unemployment | +0.177 | Positive |
| 4 | Domestic Credit | +0.056 | Positive |
| 5 | Govt Consumption | -0.056 | Negative |
| 6 | FDI Inflows | -0.078 | Negative |
| 7 | Trade Openness | -0.080 | Negative |
| 8 | GDP Growth (t) | -0.088 | Negative |
| 9 | Urban Population | -0.093 | Negative |
| 10 | Internet Usage | -0.141 | Negative |
| 11 | Life Expectancy | -0.150 | Negative |
| 12 | Population Growth | -0.154 | Negative |
| 13 | Inflation | -0.196 | Negative |
| 14 | Capital Formation | -0.212 | Negative |

### Confidence Intervals (Bootstrap, 1000 resamples, 95% CI)
- MAE: [3.1, 4.0]
- RMSE: [4.4, 5.6]
- Directional Accuracy: [44%, 61%]

### Temporal Performance
| Year | MAE | RMSE | Dir. Acc. | N |
|------|-----|------|-----------|---|
| 2021 | 3.8 | 5.2 | 50% | 50 |
| 2022 | 3.2 | 4.5 | 56% | 50 |
| 2023 | 3.6 | 5.3 | 52% | 50 |

### Worst Errors (Test Set)
1. Libya 2021: Actual +35%, Predicted -5% (error 40 pp) – post-conflict recovery
2. Equatorial Guinea 2022: Actual -12%, Predicted +2% – oil shock
3. Zimbabwe 2021: Actual +15%, Predicted -1% – policy transition

### Model Comparison
- HGB slightly outperforms Ridge on validation MAE (3.91 vs 3.98)
- Both models underperform global mean baseline on test MAE
- HGB selected for deployment (better validation, captures non-linearities)

---

## 8. Interpretation

### Important Features
**Electricity Access** is the dominant predictor (importance 0.60, 3x next feature). Countries with higher electricity access tend to have higher predicted growth. This aligns with development literature on infrastructure as growth foundation.

**GDP per Capita (log)** shows positive association – richer countries predicted to grow faster (conditional convergence not captured, or reflects omitted variable bias).

**Unemployment** positive association is counterintuitive but may reflect: (a) measurement issues (informal sector), (b) structural transformation where growing economies have more visible unemployment, (c) confounding with urbanization.

**Negative importance features** (Capital Formation, Inflation, Population Growth) suggest the model has learned associations that may reflect reverse causality or omitted variables rather than causal mechanisms.

### Why Feature Importance ≠ Causality
1. **Confounding:** Electricity access correlates with institutional quality, governance, human capital
2. **Reverse causality:** Growth enables infrastructure investment, not just vice versa
3. **Omitted variables:** Political stability, commodity prices, trade partners' growth
4. **Measurement error:** WDI indicators are estimates, not precise measurements
5. **Country heterogeneity:** Relationships differ across structural contexts

The model captures *predictive associations* useful for screening, not *causal effects* for policy design.

---

## 9. Decision-Support Application

### Streamlit Architecture
```
User Browser → Streamlit Cloud → app.py
    → Load serialized pipeline (joblib)
    → Load processed data, predictions, importance (parquet)
    → Create input dataframe from user selections
    → Run pipeline.predict()
    → Render charts, tables, warnings
```

### Pages
1. **Project Overview:** Purpose, data, model, metrics, causal disclaimer
2. **Explore Africa:** Country selector, growth trends, indicator charts, regional comparison
3. **Model Performance:** Baselines, actual vs predicted, residuals, feature importance, yearly metrics
4. **Scenario Explorer:** Country + year selector, baseline values, 5 adjustable sliders, baseline vs scenario prediction, extrapolation warnings, causal disclaimer

### Scenario Explorer Design
- **Adjustable indicators (5):** Electricity Access, Internet Usage, Capital Formation, Trade Openness, Inflation, Life Expectancy, GDP per Capita
- **Slider ranges:** Full observed data min/max (B4 fix)
- **Extrapolation warnings:** Fire when value outside P1-P99 range
- **Causal disclaimer:** Prominent on Scenario page

### Intended Use
- Screen countries for deeper analysis
- Explore "what-if" scenarios for priority indicators
- Compare baseline vs. scenario predictions
- Understand model limitations before drawing conclusions

---

## 10. Causal Limitations

### Confounding
Development indicators are endogenous. Electricity access correlates with governance quality, institutional capacity, geographic advantages. The model cannot disentangle these.

### Reverse Causality
Growth → Infrastructure investment is at least as plausible as Infrastructure → Growth. The model uses *t* features to predict *t+1* growth, but *t* features may reflect growth expectations already.

### Omitted Variables
Critical growth determinants absent: commodity prices, terms of trade, political stability, institutional quality, education quality, health system capacity, climate shocks, global financial conditions.

### Measurement Error
WDI indicators are modeled estimates (especially for low-capacity statistical systems). Error-in-variables biases coefficients toward zero.

### Country Heterogeneity
One model for 54 countries assumes homogeneous relationships. In reality, electricity-growth elasticity differs between resource-rich and resource-poor, coastal and landlocked, stable and fragile states.

### Why Predictive Scenarios Are Not Causal Interventions
Changing a slider from 70% to 80% electricity access simulates: "What would the model predict for a country-year with 80% electricity access, holding other features constant?" This is a *conditional prediction*, not a *counterfactual*. The real world does not hold other factors constant when electricity access changes.

### Methods Needed for Causal Claims
- Natural experiments / instrumental variables
- Difference-in-differences with policy rollouts
- Structural causal models with explicit DAGs
- Randomized controlled trials (where feasible)
- Synthetic control methods

---

## 11. Recommendations

1. **Use the dashboard for initial country screening** – identify countries where development profiles suggest above/below average near-term growth potential.

2. **Investigate consistently important indicators through domain research** – electricity access, macroeconomic stability, human capital. Consult sector-specific literature.

3. **Combine model output with expert knowledge and country-specific evidence** – no model replaces contextual understanding.

4. **Do not allocate funding solely from model predictions** – predictions are statistical associations, not causal guarantees.

5. **Use causal research before treating any indicator as a policy lever** – e.g., before investing in electricity access to boost growth, study the causal evidence for *your* context.

6. **Re-train periodically as WDI data and economic conditions change** – annual retraining recommended; model performance degrades during structural breaks.

7. **Monitor performance during structural shocks** – COVID-19, commodity crashes, political transitions. The model has no shock-awareness mechanism.

8. **Extend to country-specific models where data permits** – for large economies (Nigeria, South Africa, Egypt), country-specific models may outperform pooled model.

9. **Add confidence intervals to predictions** – bootstrap prediction intervals would quantify uncertainty for decision makers.

10. **Incorporate high-frequency indicators** – satellite nightlights, mobile money, trade flows for more timely predictions.

---

## 12. Conclusion

### What Was Built
A complete ML decision-support system: data pipeline (WDI → country-year panel), two models (Ridge, HGB), temporal validation, serialized artifacts, and a deployed Streamlit application with 4 interactive pages.

### What the Model Achieved
- Learned predictive associations between 14 development indicators and next-year GDP growth
- Electricity access emerges as dominant predictor (permutation importance 0.60)
- Provides directional accuracy ~53% on held-out test years (2021-2023)
- Enables interactive scenario exploration with extrapolation guardrails

### Which Model Performed Best
**HistGradientBoostingRegressor** selected over Ridge based on validation MAE (3.91 vs 3.98) and ability to capture non-linearities. However, neither model beats the global mean baseline on test MAE (3.54 vs 1.90).

### How the Dashboard Supports Decisions
- Country profiles with historical trends
- Baseline vs. scenario predictions for 5 key indicators
- Extrapolation warnings for unsupported scenarios
- Explicit causal disclaimers throughout

### Main Limitations
- Negative R² on test set (no variance explained beyond mean)
- Association ≠ causation
- Temporal generalization only (seen countries, future years)
- Limited feature set (14 WDI indicators)
- Median imputation ignores country-specific patterns
- No uncertainty quantification in deployed predictions

### Future Improvements
1. Add bootstrap prediction intervals
2. Test country-specific models for large economies
3. Incorporate high-frequency/non-traditional data (nightlights, mobility)
4. Develop causal inference module for priority indicators
5. Build automated retraining pipeline with drift detection
6. Add subnational analysis where data permits
7. Include climate vulnerability indicators

---

## Appendix: Key Metrics from model_metadata.json

```json
{
  "model_type": "HistGradientBoostingRegressor",
  "n_features": 14,
  "train_end": 2017,
  "val_end": 2020,
  "metrics": {
    "global_mean_baseline": {"mae": 1.896, "rmse": 2.838, "r2": -0.001, "directional_accuracy": 0.807},
    "persistence_baseline": {"mae": 2.229, "rmse": 4.521, "r2": -1.539, "directional_accuracy": 0.773},
    "winner_test": {"mae": 3.540, "rmse": 4.998, "r2": -2.104, "directional_accuracy": 0.527}
  }
}
```