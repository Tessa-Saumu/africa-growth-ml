## Project Title

**Africa Growth Explorer: A Machine Learning Decision-Support System Using World Bank Development Indicators**

### Alternative Shorter Dashboard Title

**Africa Growth Explorer**

---

# 1. Project Purpose

## Core Project Question

> To what extent can recent development indicators predict near-term GDP per capita growth across African countries, and which observed development conditions are most informative for those predictions?

## Decision-Support Question

> Given a country’s current development profile, what level of next-year GDP per capita growth does the model estimate, which indicators contribute most to that estimate, and how does the estimate change under alternative development scenarios?

## Intended User

The system is designed for:

- Development analysts
- Economic researchers
- Policy analysts
- Government planning teams
- NGOs and development institutions
- Students and researchers comparing development conditions across African countries

The application is **not intended to make final policy decisions**. It is a screening and analytical tool to support deeper investigation.

---

# 2. Important Interpretation Boundary

The project will use machine learning for **prediction and decision support**, not causal policy-effect estimation.

The model can identify that certain development indicators are associated with higher or lower predicted future growth.

It cannot prove that changing one indicator will cause a particular increase in GDP growth.

For example:

> If electricity access is increased from 70% to 80% in the scenario explorer and the model prediction increases, this means the model associates that feature profile with higher predicted growth. It does not prove that increasing electricity access alone will cause the predicted increase.

This distinction should appear in:

- The project report
- The README
- The Streamlit application
- The presentation slides, if you create them

This demonstrates understanding of the difference between:

- Prediction
- Association
- Causality
- Intervention effects

---

# 3. Project Scope

## Geographic Scope

African countries only.

The African country list should be created once in the data-processing pipeline and reused everywhere.

### Preferred Approach

1. Use World Bank country metadata if available.
2. Filter using the World Bank region classification.
3. Exclude aggregates such as:
   - Sub-Saharan Africa
   - Africa Eastern and Southern
   - Africa Western and Central
   - World
   - Income groups
   - Regional aggregates

If metadata is inconvenient, maintain an explicit ISO3 country-code list in the configuration file.

Use ISO3 codes internally rather than relying only on country names.

## Time Scope

Use all available years with sufficient coverage, subject to the following design:

- Features observed at year `t`
- Target observed at year `t + 1`

Example:

```text
Country: Ghana
Feature year: 2018
Target year: 2019
````

The exact final start and end years should be determined after the data audit.

A reasonable initial range is:

```text
2000–latest usable year
```

The final target year will be one year earlier than the latest available GDP per capita growth observation.

---

# 4. Dataset

## Required Source

World Bank World Development Indicators.

Source:

[https://datatopics.worldbank.org/world-development-indicators/](https://datatopics.worldbank.org/world-development-indicators/)

The project will use only WDI data, satisfying the assignment requirement.

## Raw Input

The downloaded WDI CSV archive:

```text
WDI_CSV.zip
```

The raw data should not be committed to GitHub if it is too large.

Instead, include:

- Source URL
    
- Download instructions
    
- Expected file names
    
- Data-processing instructions
    
- A small processed dataset or Git LFS/reference approach if necessary
    

## Main WDI Target

Use:

```text
NY.GDP.PCAP.KD.ZG
```

Indicator:

```text
GDP per capita growth (annual %)
```

## Prediction Target

For each country-year:

```text
target = GDP per capita growth in year t + 1
```

The target should be created using a grouped country shift:

```python
df["target_next_year"] = (
    df.sort_values(["iso3", "year"])
      .groupby("iso3")["gdp_per_capita_growth"]
      .shift(-1)
)
```

The final year for each country will normally have no target and must be removed from the training dataset.

---

# 5. Candidate Features

Begin with the following candidate features. The exact final feature set should be selected after measuring data coverage.

|Feature|WDI Indicator Code|Theme|
|---|---|---|
|GDP per capita growth|`NY.GDP.PCAP.KD.ZG`|Economic performance|
|GDP per capita|`NY.GDP.PCAP.CD` or constant-price equivalent|Economic level|
|Access to electricity|`EG.ELC.ACCS.ZS`|Infrastructure|
|Individuals using the Internet|`IT.NET.USER.ZS`|Technology|
|Gross capital formation|`NE.GDI.TOTL.ZS`|Investment|
|Foreign direct investment, net inflows|`BX.KLT.DINV.WD.GD.ZS`|External investment|
|Trade|`NE.TRD.GNFS.ZS`|Openness|
|Inflation|`FP.CPI.TOTL.ZG`|Macroeconomic stability|
|Unemployment|`SL.UEM.TOTL.ZS`|Labour market|
|Life expectancy at birth|`SP.DYN.LE00.IN`|Health|
|Secondary school enrollment|Relevant WDI code after validation|Education|
|Domestic credit to private sector|`FS.AST.PRVT.GD.ZS`|Financial development|
|Government consumption|`NE.CON.GOVT.ZS`|Public sector|
|Urban population|`SP.URB.TOTL.IN.ZS`|Demographics|
|Population growth|`SP.POP.GROW`|Demographics|

The exact education indicator code must be validated from the downloaded WDI metadata because WDI provides multiple enrollment definitions and coverage differs.

## Feature Selection Rules

Retain indicators that satisfy most of the following:

- At least approximately 60–70% usable country-year coverage.
    
- Conceptually relevant to growth.
    
- Available before the prediction year.
    
- Not directly derived from the target.
    
- Not a duplicate of another retained indicator.
    
- Sufficiently interpretable for the dashboard.
    

The likely final feature count is approximately:

```text
8–12 features
```

**Do not force all candidate features into the model.**

---

# 6. Data Representation

The raw WDI file is indicator-oriented and wide.

Example raw structure:

```text
Country Name | Country Code | Indicator Name | Indicator Code | 2000 | 2001 | ...
```

Transform it into a country-year panel:

```text
iso3 | country_name | year | feature_1 | feature_2 | ... | target_next_year
```

Example:

```text
GHA | Ghana | 2018 | 78.4 | 41.7 | 23.1 | 6.2
```

## Required Processing Sequence

```text
Raw WDI CSV
    ↓
Select required columns
    ↓
Filter African countries
    ↓
Filter selected indicator codes
    ↓
Reshape from wide to long
    ↓
Pivot indicators into feature columns
    ↓
Convert year columns to numeric
    ↓
Sort by country and year
    ↓
Create next-year target
    ↓
Apply missing-data rules
    ↓
Create temporal train/validation/test datasets
```

---

# 7. Data Cleaning Rules

All cleaning decisions should be documented in the report.

## Duplicates

- Check for duplicate `(iso3, year)` rows.
    
- Remove exact duplicates.
    
- Investigate conflicting duplicate records rather than silently dropping them.
    

## Numeric Conversion

- Convert WDI year columns to numeric.
    
- Convert blank strings and placeholder values to `NaN`.
    
- Ensure all model features are numeric.
    

## Missing Values

First calculate:

- Missingness by indicator
    
- Missingness by country
    
- Missingness by year
    

### Recommended Policy

1. Drop features with extremely poor coverage.
    
2. Drop rows where the target is missing.
    
3. Use model-pipeline median imputation for remaining feature missingness.
    
4. Do not globally fill values using information from future years.
    
5. Do not forward-fill volatile variables such as inflation or FDI without strong justification.
    

The imputer must be fitted only on the training data through the `sklearn` pipeline.

## Outliers

Do not automatically remove all statistical outliers.

Extreme observations may represent genuine events such as:

- Economic crises
    
- Hyperinflation
    
- Commodity shocks
    
- Major FDI inflows
    
- Recessions
    
- Pandemic effects
    

### Required Procedure

- Inspect extreme values.
    
- Confirm that they are not parsing errors.
    
- Retain legitimate observations.
    
- Consider transformations for heavily skewed features.
    
- Report their effect on model performance if practical.
    

## Transformations

Possible transformations:

- Log transformation for GDP per capita levels.
    
- Standardization for Ridge.
    
- Optional robust clipping for exceptionally unstable predictor values.
    

Do not transform the target unless necessary. Keep GDP per capita growth in percentage points for interpretability.

---

# 8. Problem Formulation

## ML Task

Regression.

## Observation Unit

One African country-year.

## Input

Development indicators observed during year `t`.

## Output

Predicted GDP per capita growth during year `t + 1`.

## Example

```text
Input:
Kenya's development indicators in 2019

Output:
Predicted GDP per capita growth in 2020
```

## Why This Is an Appropriate ML Problem

The relationship between development conditions and future growth is likely:

- Non-linear
    
- Multi-dimensional
    
- Country-dependent
    
- Influenced by interactions between indicators
    

This makes a predictive model useful as a supplement to descriptive analytics and fixed rules.

---

# 9. Validation Strategy

## Why Random Splitting Is Not Appropriate

Country-year observations are ordered in time and related within countries.

A random split could allow information from later years to influence training while earlier years are used for evaluation. This would provide an overly optimistic estimate of real-world performance.

## Main Split

Use a chronological split.

Initial structure:

```text
Training:   2000–2017
Validation: 2018–2020
Test:       2021–latest usable year
```

The exact years should be finalized after checking the data.

The test set must remain untouched until:

- Features are finalized.
    
- Models are selected.
    
- Hyperparameters are selected.
    
- The decision policy is defined.
    

## Hyperparameter Tuning

Use a small expanding-window validation strategy inside the training period.

Example:

```text
Train 2000–2010 → validate 2011–2012
Train 2000–2012 → validate 2013–2014
Train 2000–2014 → validate 2015–2016
```

Do not perform a large search. Use a compact grid or a small randomized search.

## Generalization Interpretation

This project evaluates whether models can predict future years for African countries with historical observations in the dataset.

It does not primarily evaluate prediction for completely unseen countries.

This should be explicitly stated in the report.

---

# 10. Baselines

Use at least two baselines where practical.

## Baseline 1: Global Mean

Predict the average training-period target for every test observation.

This is a basic reference point.

## Baseline 2: Persistence

Predict next year's growth as this year's growth.

```text
Predicted growth at t+1 = observed growth at t
```

This is a meaningful time-series baseline.

## Optional Baseline 3: Country Historical Mean

Predict using the country's historical average growth up to the prediction point.

This is useful but should only be included if implemented without using future information.

The ML models must be compared against these baselines.

---

# 11. Machine Learning Models

The assignment requires at least two models.

## Model 1: Ridge Regression

Pipeline:

```text
MedianImputer
    ↓
StandardScaler
    ↓
Ridge Regression
```

### Purpose

- Strong linear benchmark.
    
- Robust to correlated indicators.
    
- Easier to interpret.
    
- Establishes whether a simple regularized model is sufficient.
    

## Model 2: Gradient Boosting Regressor

Preferred implementation:

```text
HistGradientBoostingRegressor
```

Alternative:

```text
XGBRegressor
```

Use `HistGradientBoostingRegressor` if you want fewer deployment dependencies.

### Purpose

- Captures non-linear relationships.
    
- Captures interactions between features.
    
- Provides a stronger tabular-data benchmark.
    

## Model Selection Principle

Select the final model using:

1. Validation MAE.
    
2. Validation RMSE.
    
3. Stability across temporal folds.
    
4. Test-set performance.
    
5. Interpretability and deployment simplicity.
    

**Do not choose a model based only on the highest R².**

---

# 12. Evaluation Metrics

## Primary Metric

### Mean Absolute Error

```text
MAE
```

Interpretation:

> On average, the model's prediction differs from actual GDP per capita growth by approximately X percentage points.

## Secondary Metrics

### Root Mean Squared Error

```text
RMSE
```

This penalizes large prediction errors more heavily.

### R²

Use as a supplementary measure of explained variation.

Do not present R² as the only measure of model quality.

### Directional Accuracy

Convert actual and predicted growth into positive/negative movement:

```text
growth >= 0 → positive/non-negative
growth < 0 → negative
```

Then calculate the proportion of cases where the model correctly identifies the direction.

This aligns the model with a practical screening use case.

## Additional Analyses

Calculate:

- Metrics by year
    
- Metrics by country
    
- Metrics by subregion if feasible
    
- Error distribution
    
- Worst prediction errors
    
- Performance during major shock years
    
- Actual vs. predicted plot
    
- Residual plot
    

## Confidence Intervals

If time permits, calculate bootstrap 95% confidence intervals for:

- MAE
    
- RMSE
    
- Directional accuracy
    

Use bootstrap resampling on the test predictions.

If time becomes limited, prioritize correct temporal validation over bootstrap intervals.

---

# 13. Model Interpretation

## Global Interpretation

Use:

- Ridge standardized coefficients
    
- Permutation importance for the gradient boosting model
    

Show:

- Which indicators contribute most to predictive performance.
    
- Whether the relationship is directionally positive or negative for Ridge.
    
- Differences between linear and non-linear model interpretations.
    

## Optional SHAP

SHAP can be included if it works cleanly and does not delay deployment.

It is not mandatory.

If implemented, use:

- Global summary plot
    
- Local explanation for a selected country
    
- Feature contribution bar chart
    

If SHAP creates dependency or compatibility problems, replace it with:

- Permutation importance
    
- Scenario prediction changes
    
- Ridge coefficients
    

The portfolio value comes from defensible interpretation, not from using a specific interpretability package.

---

# 14. Scenario Explorer Design

The scenario explorer is the main decision-support feature.

## User Flow

1. Select an African country.
    
2. Select a reference year.
    
3. View the country's observed indicators.
    
4. View the model's baseline prediction for next-year GDP per capita growth.
    
5. Adjust selected scenario indicators.
    
6. Generate a new prediction.
    
7. Compare baseline and scenario predictions.
    
8. Review the warning about predictive association and causality.
    

## Scenario Variables

Use 3–5 variables with understandable units and adequate coverage, for example:

- Electricity access
    
- Internet usage
    
- Gross capital formation
    
- Trade openness
    
- Inflation
    
- Life expectancy
    
- GDP per capita level
    

Do not allow users to modify every feature. Too many sliders reduce interpretability.

## Output

Example:

```text
Country: Ghana
Reference year: 2019

Baseline predicted next-year growth: 3.1%
Scenario predicted next-year growth: 3.8%
Model-implied difference: +0.7 percentage points
```

## Required Wording

Display prominently:

> Scenario results show how the predictive model responds to alternative indicator values. They should not be interpreted as causal estimates of the effect of implementing a specific policy.

## Scenario Guardrails

For every adjustable feature:

- Define observed training-data minimum and maximum.
    
- Warn when the user enters an extreme or unsupported value.
    
- Prefer percentile-based warnings, such as values outside the 1st–99th percentile.
    
- Warn when the scenario differs substantially from observed African country-year profiles.
    

Example:

> **Warning:** This scenario is near or outside the range of historical values observed by the model. The result may be unreliable because it represents extrapolation.

---

# 15. Streamlit Application

## Deployment Approach

Use **Streamlit Cloud only**.

Do not use:

- FastAPI
    
- Flask
    
- Render
    
- Separate frontend/backend services
    
- Live API calls to a backend
    

The Streamlit application will perform inference directly on the Streamlit server.

## Runtime Architecture

```text
User browser
    ↓
Streamlit Cloud
    ↓
app.py
    ↓
Load serialized sklearn pipeline
    ↓
Create input dataframe
    ↓
Run model.predict()
    ↓
Render prediction, explanation, and charts
```

This avoids the previous problem where the frontend was deployed but the backend was unavailable.

## Application Pages or Tabs

### 1. Project Overview

Include:

- Project purpose
    
- Problem statement
    
- Data source
    
- Target definition
    
- Geographic scope
    
- Model limitations
    
- Causal interpretation warning
    

### 2. Explore Africa

Include:

- Country selector
    
- Growth trend chart
    
- Selected indicator trends
    
- Latest available comparison table
    
- Basic descriptive statistics
    

### 3. Model Performance

Include:

- Baseline comparison
    
- Model metrics
    
- Actual vs. predicted chart
    
- Residual chart
    
- Feature importance
    
- Model limitations
    

### 4. Scenario Explorer

Include:

- Country selector
    
- Year selector
    
- Current feature values
    
- Scenario controls
    
- Baseline prediction
    
- Scenario prediction
    
- Prediction difference
    
- Support/extrapolation warning
    
- Causal interpretation disclaimer
    

## Streamlit Technical Requirements

Use:

```python
@st.cache_resource
```

for model loading.

Use:

```python
@st.cache_data
```

for processed-data loading.

The app must load artifacts rather than retraining models.

No live WDI API call should be required for the app to work.

---

# 16. Production-Oriented Repository

Recommended structure:

```text
africa-growth-ml/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── config/
│   └── indicators.yaml
│
├── data/
│   ├── README.md
│   └── processed/
│       ├── model_data.parquet
│       └── country_metadata.csv
│
├── models/
│   ├── growth_model.joblib
│   ├── preprocessing_metadata.json
│   └── model_metadata.json
│
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_model_evaluation.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   └── visualization.py
│
└── reports/
    └── capstone_report.pdf
```

## Purpose of Each Source File

### `src/data.py`

- Load raw WDI CSV.
    
- Load metadata.
    
- Filter African countries.
    
- Select indicators.
    
- Reshape data.
    
- Validate structure.
    
- Log data dimensions.
    

### `src/features.py`

- Create country-year panel.
    
- Create lagged/current features.
    
- Create next-year target.
    
- Apply feature selection.
    
- Generate training-ready data.
    

### `src/train.py`

- Define baselines.
    
- Define sklearn pipelines.
    
- Train models.
    
- Tune model parameters.
    
- Save final artifact.
    
- Save metadata.
    

### `src/evaluate.py`

- Calculate regression metrics.
    
- Calculate directional accuracy.
    
- Bootstrap confidence intervals.
    
- Generate error analysis.
    
- Calculate feature importance.
    

### `src/visualization.py`

- Reusable EDA charts.
    
- Performance charts.
    
- Trend charts.
    
- Feature importance charts.
    

### `app.py`

- Load artifacts.
    
- Render Streamlit pages.
    
- Accept user inputs.
    
- Run predictions.
    
- Display results and warnings.
    

---

# 17. Model Artifact Requirements

Save one complete fitted pipeline rather than separate preprocessing components.

Example:

```python
joblib.dump(final_pipeline, "models/growth_model.joblib")
```

The pipeline should contain:

```text
Imputer
→ Scaler, where required
→ Model
```

Save metadata separately:

```json
{
  "target": "GDP per capita growth (annual %)",
  "target_code": "NY.GDP.PCAP.KD.ZG",
  "prediction_horizon_years": 1,
  "geographic_scope": "African countries",
  "features": [
    "electricity_access",
    "internet_usage"
  ],
  "training_years": "2000-2017",
  "validation_years": "2018-2020",
  "test_years": "2021-2023",
  "model_type": "HistGradientBoostingRegressor",
  "mae": null,
  "rmse": null,
  "r2": null,
  "random_seed": 42
}
```

The actual values should be populated after training.

---

# 18. Logging and Reproducibility

Use a fixed random seed, for example:

```python
RANDOM_STATE = 42
```

Log:

- Raw file path
    
- Raw dataset shape
    
- Number of African countries
    
- Selected indicator codes
    
- Dropped indicators and reasons
    
- Missingness percentages
    
- Number of final rows
    
- Train/validation/test periods
    
- Model parameters
    
- Evaluation metrics
    
- Artifact save paths
    

Keep a `logs/` directory locally, but do not necessarily commit large log files.

---

# 19. Report Structure

The PDF report should contain the following sections.

## 1. Introduction

Explain:

- Economic growth and development context.
    
- Why African development analysis is important.
    
- Why decision support is more useful than isolated prediction.
    
- Role of WDI.
    

## 2. Problem Statement

Clearly state:

- Prediction task.
    
- Target.
    
- Time horizon.
    
- Geographic scope.
    
- Intended user.
    
- Expected impact.
    

## 3. Dataset Description

Include:

- World Bank WDI source.
    
- Number of indicators considered.
    
- Number of countries.
    
- Time range.
    
- Final feature set.
    
- Target definition.
    
- Data limitations.
    

## 4. Methodology

Explain:

- Data transformation.
    
- Missing-data strategy.
    
- Outlier handling.
    
- Feature engineering.
    
- Temporal split.
    
- Baselines.
    
- Models.
    
- Evaluation metrics.
    

## 5. Exploratory Data Analysis

Include:

- Summary statistics.
    
- Missingness.
    
- Trend analysis.
    
- Distributions.
    
- Correlations.
    
- Country comparisons.
    
- Main EDA findings.
    

## 6. Model Development

Explain:

- Ridge Regression.
    
- Gradient Boosting.
    
- Pipeline design.
    
- Hyperparameter tuning.
    
- Reproducibility.
    

## 7. Model Evaluation

Include a table like:

|Model|MAE|RMSE|R²|Directional Accuracy|
|---|--:|--:|--:|--:|
|Global mean baseline|||||
|Persistence baseline|||||
|Ridge Regression|||||
|Gradient Boosting|||||

Also include:

- Actual vs. predicted chart.
    
- Residual analysis.
    
- Confidence intervals if completed.
    
- Model comparison.
    
- Temporal performance.
    
- Worst errors.
    

## 8. Interpretation

Discuss:

- Important features.
    
- Model coefficients or permutation importance.
    
- Country-specific interpretation.
    
- Why feature importance does not imply causality.
    

## 9. Decision-Support Application

Explain:

- Streamlit architecture.
    
- Country selection.
    
- Baseline prediction.
    
- Scenario exploration.
    
- Extrapolation warnings.
    
- Intended use.
    

## 10. Causal Limitations

Discuss:

- Confounding.
    
- Reverse causality.
    
- Omitted variables.
    
- Measurement error.
    
- Country heterogeneity.
    
- Why predictive scenarios are not causal interventions.
    
- What methods would be needed for causal claims.
    

## 11. Recommendations

Recommendations should be practical and appropriately cautious.

Examples:

- Use the dashboard for initial country screening.
    
- Investigate consistently important indicators through domain research.
    
- Combine model output with expert knowledge and country-specific evidence.
    
- Do not allocate funding solely from model predictions.
    
- Use causal research before treating any indicator as a policy lever.
    
- Re-train periodically as WDI data and economic conditions change.
    
- Monitor performance during structural shocks.
    

## 12. Conclusion

Summarize:

- What was built.
    
- What the model achieved.
    
- Which model performed best.
    
- How the dashboard supports decisions.
    
- Main limitations.
    
- Future improvements.
    

---

# 20. README Requirements

The README should include:

1. Project title.
    
2. Project overview.
    
3. Business/social/economic problem.
    
4. Key question.
    
5. Dataset source.
    
6. Indicator and target definitions.
    
7. Project architecture.
    
8. Setup instructions.
    
9. How to run the data pipeline.
    
10. How to run the Streamlit app.
    
11. Model evaluation summary.
    
12. Deployment link.
    
13. Screenshot or GIF of the dashboard.
    
14. Important limitations.
    
15. Causal interpretation disclaimer.
    
16. Folder structure.
    
17. Reproducibility instructions.
    

Example run command:

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

# 21. Requirements File

Keep dependencies modest.

Likely requirements:

```text
pandas
numpy
scikit-learn
matplotlib
seaborn
plotly
streamlit
joblib
pyarrow
pyyaml
```

Add `xgboost` only if you decide to use XGBoost instead of sklearn's histogram gradient boosting.

Avoid unnecessary dependencies that could create Streamlit Cloud build failures.

---

# 22. Two-Day Implementation Schedule

## Day 1: Data, Modeling, Evaluation

### Morning

- Create repository.
    
- Download and inspect WDI.
    
- Identify WDI files and metadata.
    
- Validate African-country filtering.
    
- Validate indicator codes.
    
- Calculate coverage.
    
- Finalize features.
    
- Build the country-year panel.
    
- Create the next-year target.
    

### Afternoon

- Implement cleaning and feature pipeline.
    
- Create EDA notebook.
    
- Implement temporal train/validation/test split.
    
- Build global mean and persistence baselines.
    
- Train Ridge.
    
- Train Gradient Boosting.
    
- Tune lightly.
    

### Evening

- Evaluate models.
    
- Generate metrics table.
    
- Generate actual-vs-predicted plot.
    
- Generate residual plots.
    
- Generate permutation importance.
    
- Inspect worst errors.
    
- Select final model.
    
- Save final pipeline and metadata.
    

### Day 1 Acceptance Condition

Before ending Day 1, you must have:

- A processed dataset.
    
- A complete trained pipeline.
    
- A saved `.joblib` artifact.
    
- A metrics table.
    
- A basic prediction script that works on new input rows.
    

**Do not delay model training until after the dashboard.**

---

## Day 2: Application, Deployment, Report

### Morning

- Build Streamlit app.
    
- Implement model loading.
    
- Implement country and year selection.
    
- Implement historical charts.
    
- Implement model performance tab.
    
- Implement scenario explorer.
    
- Implement range/extrapolation warnings.
    

### Midday

- Deploy to Streamlit Cloud immediately.
    
- Confirm that the deployed application loads the model.
    
- Test actual predictions.
    
- Fix dependency and path issues.
    

### Afternoon

- Improve layout and labels.
    
- Add clear limitations.
    
- Complete README.
    
- Write report.
    
- Add screenshots.
    
- Create final GitHub repository.
    
- Test from a clean environment.
    
- Submit the Streamlit link and repository link.
    

---

# 23. Deployment Checklist

Before deployment, confirm:

-  `app.py` is at the repository root.
    
-  All imports work from the repository root.
    
-  Model files are committed or otherwise available.
    
-  Processed data files are available.
    
-  No local absolute paths are used.
    
-  No API server is required.
    
-  No local-only environment variables are required.
    
-  `requirements.txt` installs successfully.
    
-  The app works with a fresh Streamlit Cloud environment.
    
-  The app does not retrain the model at runtime.
    
-  The app can make a prediction after selecting a country.
    
-  Scenario predictions update successfully.
    
-  Warnings display for unsupported scenarios.
    
-  The causal disclaimer is visible.
    

---

# 24. Portfolio-Readiness Checklist

The final project will be portfolio-ready if it demonstrates the following.

## Problem Maturity

-  Clear real-world user.
    
-  Clear prediction target.
    
-  Clear decision-support application.
    
-  Explicit business/social value.
    

## Data Maturity

-  Real World Bank dataset.
    
-  Documented indicator selection.
    
-  Country-year panel construction.
    
-  Missingness analysis.
    
-  Defensible cleaning decisions.
    
-  No unexplained data manipulation.
    

## Modeling Maturity

-  Meaningful baselines.
    
-  At least two models.
    
-  Leakage-aware temporal validation.
    
-  Appropriate regression metrics.
    
-  Error analysis.
    
-  Model selection based on evidence.
    

## Decision-System Maturity

-  Predictions are converted into understandable outputs.
    
-  Scenario exploration is available.
    
-  Users can compare baseline and alternative profiles.
    
-  Unsupported scenarios receive warnings.
    
-  The system explains what the model can and cannot claim.
    

## Engineering Maturity

-  Reusable source modules.
    
-  Lean notebooks.
    
-  Serialized model artifact.
    
-  Centralized configuration.
    
-  Logging.
    
-  Reproducible setup.
    
-  Streamlit deployment with live inference.
    
-  No disconnected backend.
    

## Communication Maturity

-  Professional report.
    
-  Clean dashboard.
    
-  Clear README.
    
-  Visual model comparison.
    
-  Explicit limitations.
    
-  Causal interpretation handled responsibly.
    

---

# 25. Requirements Mapping to the Internship Brief

|Internship Requirement|How This Project Satisfies It|
|---|---|
|Define a real-world problem|Predicting near-term GDP per capita growth to support African development analysis|
|Use WDI dataset|All modeling and analysis use World Bank WDI|
|Collect and understand data|WDI download, metadata inspection, indicator coverage analysis|
|Data cleaning|Missing values, duplicates, numeric conversion, country filtering, outlier investigation|
|Feature engineering|Country-year panel and next-year target construction|
|EDA|Distributions, trends, correlations, country comparisons, missingness|
|Build at least two ML models|Ridge Regression and Gradient Boosting|
|Evaluate models|MAE, RMSE, R², directional accuracy, baseline comparison|
|Compare models|Validation and held-out temporal test comparison|
|Generate insights|Feature importance, trend analysis, error analysis, country profiles|
|Provide recommendations|Responsible use, deeper investigation, causal evidence, monitoring|
|Deploy a model|Streamlit Cloud application with local serialized-model inference|
|Submit report|Structured PDF report|
|Submit source code|GitHub repository with reusable `src/` modules and notebooks|
|Include dataset source|WDI source URL and download documentation|
|Presentation requirement|The brief allows a presentation or demo video; if a live presentation is required, prepare concise slides. No video is needed unless specifically requested by the program.|

### Presentation Clarification

The brief says:

> “Prepare a 5–10 minute presentation or demo video.”

If “presentation” means a live or slide-based presentation, you should prepare a short slide deck.

You do **not** need to record a video unless instructed separately by the program.

---

# 26. Final Scope Decisions

The following decisions are now fixed:

- Direction A.
    
- Africa-only analysis.
    
- GDP per capita growth prediction.
    
- One-year-ahead prediction horizon.
    
- World Bank WDI only.
    
- Temporal validation rather than random splitting.
    
- Ridge Regression and Gradient Boosting.
    
- Baselines included.
    
- Streamlit Cloud deployment.
    
- Direct in-app model inference.
    
- No FastAPI or Render.
    
- No live backend.
    
- Medium-depth causal discussion.
    
- Interactive scenario explorer.
    
- Scenario guardrails.
    
- Lean notebooks.
    
- Reusable source modules.
    
- Serialized model artifact.
    
- Logging and reproducibility.
    
- No video unless separately required.
    

The only implementation decisions that should remain data-dependent are:

- Final feature list.
    
- Exact year split.
    
- Final winning model.
    
- Whether SHAP and prediction intervals can be added without threatening deployment.
    

---

# Final Recommendation

This is the correct scope for your two-day deadline.

It fulfills the internship brief while presenting you as someone who understands ML as a **complete decision-support system**, rather than simply as a model-training exercise.

```

### Obsidian-specific changes I made

- Standardized heading levels.
- Converted the deployment and portfolio sections into actual Obsidian checklists.
- Preserved all Python, JSON, YAML, shell, and text examples as fenced code blocks.
- Used backticks consistently for indicator codes, filenames, variables, and commands.
- Kept tables in standard GitHub/Obsidian-compatible Markdown.
- Preserved horizontal rules as `---`.
- Removed unnecessary Markdown inconsistencies while keeping your original meaning.
- Added a final `# Final Recommendation` section to give the document a clean ending.
```