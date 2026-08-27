# Africa Growth Explorer - Presentation Outline

**5-10 Minute Presentation / Demo Script**

---

## Slide 1: Title Slide (30 seconds)

**Africa Growth Explorer**
*A Machine Learning Decision-Support System Using World Bank Development Indicators*

**Core Question:** Can recent development indicators predict near-term GDP per capita growth across African countries?

**Presenter:** [Your Name]
**Date:** [Date]
**Program:** FlyRank ML Engineering Internship Capstone

---

## Slide 2: Problem & Motivation (60 seconds)

### Why This Matters
- 54 African countries, diverse growth trajectories
- Development analysts need screening tools, not just predictions
- "Ghana will grow 3.2%" is less useful than "If Ghana improves electricity access from 70%→80%, model predicts +0.7pp growth"

### Decision-Support Framing
- **Not:** Causal policy engine
- **Yes:** Statistical association explorer for hypothesis generation
- **User:** Analysts, researchers, policy teams, NGOs

---

## Slide 3: Data & Target (60 seconds)

### Source
World Bank WDI — 14 indicators, 52 countries, 2000-2024

### Target Construction
```
Features at year t  →  Target at year t+1
Ghana 2019 indicators  →  Ghana 2020 GDP growth
```

### Key Indicators (Themes)
| Infrastructure | Technology | Investment | Macro Stability | Labour | Health |
|---|---|---|---|---|---|
| Electricity Access | Internet Usage | Capital Formation | Inflation | Unemployment | Life Expectancy |
| | | FDI Inflows | | | |
| | | Domestic Credit | | | |

### Temporal Split (No Leakage!)
- **Train:** 2000-2017
- **Validation:** 2018-2020 (includes COVID shock)
- **Test:** 2021+ (held out until final evaluation)

---

## Slide 4: Methodology (60 seconds)

### Pipeline
```
Raw WDI CSV
    → Filter African countries (explicit ISO3 list, avoids MENA trap)
    → Select 14 indicators + target
    → Reshape wide → long → country-year panel
    → Create next-year target (grouped shift)
    → Coverage filter (≥60% on training rows only)
    → Temporal split
    → sklearn Pipeline: Imputer → Model
    → Serialize single joblib artifact
```

### Models
1. **Ridge Regression** — Linear benchmark, interpretable coefficients
2. **HistGradientBoostingRegressor** — Non-linear, interactions, selected for deployment

### Baselines
- Global Mean (training average)
- Persistence (current year = next year)

---

## Slide 5: Results — Model Performance (90 seconds)

### Test Set Metrics (2021-2023, 150 observations)

| Model | MAE (pp) | RMSE (pp) | R² | Dir. Acc. |
|---|---:|---:|---:|---:|
| Global Mean Baseline | 1.90 | 2.84 | -0.00 | 80.7% |
| Persistence Baseline | 2.23 | 4.52 | -1.54 | 77.3% |
| **HGB (Selected)** | **3.54** | **5.00** | **-2.10** | **52.7%** |

### Key Finding
- **Negative R²** — Model doesn't beat global mean on variance explained
- **But** — Directional accuracy 52.7% (better than random)
- **Value is in:** Feature importance + Scenario exploration, not point accuracy

### Feature Importance (Permutation, Test Set)
1. **Electricity Access** (+0.60) — 3x next feature
2. **GDP per Capita** (+0.22)
3. **Unemployment** (+0.18)
4. **Inflation** (-0.20)
5. **Capital Formation** (-0.21)

---

## Slide 6: Feature Importance Interpretation (60 seconds)

### What It Means
- **Electricity access** is the strongest *predictive associate* of next-year growth
- Higher electricity access → higher predicted growth
- **Does NOT mean** building power plants causes growth
- Confounding: governance, institutions, geography all correlate with electricity

### Causal Boundary (Critical)
> "Scenario results show how the predictive model responds to alternative indicator values. They should **not** be interpreted as causal estimates of the effect of implementing a specific policy."

### Why This Matters
- Prevents misuse: "Model says electricity → growth, so fund power plants"
- Encourages: "Model highlights electricity; let's research the causal evidence"

---

## Slide 7: Live Demo — Streamlit App (3 minutes)

### Page 1: Project Overview
- Problem statement, data, model card, causal disclaimer

### Page 2: Explore Africa
- Country selector → Ghana
- Growth trend chart (observed vs target)
- Latest indicators dashboard
- Multi-indicator trend plots
- Regional comparison table (latest year)

### Page 3: Model Performance
- Baseline comparison table
- Actual vs Predicted scatter
- Residual plot
- Feature importance chart
- Yearly metrics breakdown

### Page 4: Scenario Explorer ← **Core Decision-Support Feature**
1. Select country (Ghana) + year (2019)
2. View baseline feature values
3. Adjust sliders (5 key indicators, full data range)
4. **Extrapolation warnings** fire at P1-P99 boundaries
5. See baseline vs scenario prediction + difference
6. Causal disclaimer prominent

---

## Slide 8: Technical Rigor (60 seconds)

### Fixes Implemented
| Issue | Fix |
|---|---|
| B2: Persistence baseline wrong | Predicts current year's growth, not training mean |
| B3: Feature list drift | Single source: `model_metadata.json` |
| B4: Slider ranges | Full data min/max; warnings at P1-P99 |
| B5: Recomputing on rerun | Precomputed predictions & importance from parquet |
| B7: MENA substring trap | Explicit ISO3 country list |
| B8: Pickling lambdas | Named function in `src.features.clip_log1p` |

### Engineering Quality
- All imports work from root
- `@st.cache_resource` / `@st.cache_data` for performance
- 33 unit tests passing
- No live API calls, no absolute paths
- Python 3.11 compatible
- Deployed on Streamlit Cloud

---

## Slide 9: Limitations & Honest Assessment (60 seconds)

### What the Model Does Well
- Identifies electricity access as dominant predictive signal
- Enables rapid cross-country screening
- Scenario explorer with guardrails

### What It Doesn't Do
- Beat global mean baseline on MAE/R²
- Prove causality
- Generalize to unseen countries
- Quantify prediction uncertainty (no intervals in app)

### When to Trust / Not Trust
| Trust For | Don't Trust For |
|---|---|
| Screening countries for deeper dive | Allocating budget based on predictions |
| Exploring "what-if" associations | Designing policy interventions |
| Understanding predictive signals | Replacing expert judgment |

---

## Slide 10: Future Work (30 seconds)

1. **Prediction intervals** — Bootstrap CIs for each prediction
2. **Country-specific models** — For Nigeria, South Africa, Egypt
3. **High-frequency data** — Nightlights, mobile money, trade flows
4. **Causal module** — IV, DiD for priority indicators (electricity)
5. **Automated retraining** — Annual pipeline with drift detection
6. **Subnational analysis** — Where data permits

---

## Slide 11: Key Takeaways (30 seconds)

1. **Built a complete ML system** — Data → Model → App → Deploy
2. **Decision-support > raw prediction** — Scenario explorer is the product
3. **Causal humility** — Explicit disclaimers, not buried in appendix
4. **Engineering rigor** — Temporal splits, single source of truth, cached artifacts
5. **Honest about limits** — Negative R², baseline comparison, extrapolation warnings

---

## Slide 12: Links & Contact (15 seconds)

- **GitHub:** [github.com/yourusername/africa-growth-ml]
- **Streamlit App:** [your-app.streamlit.app]
- **Report:** `reports/capstone_report.md`
- **Contact:** [your email]

---

## Demo Script Notes (for live presentation)

### Scenario Explorer Demo Flow
1. "Let's explore Ghana in 2019"
2. "Baseline prediction: 3.1% growth"
3. "What if electricity access increases from 79% to 90%?" → slide slider
4. "Warning fires: 90% is above P99 (87%) — extrapolation"
5. "Scenario prediction: 3.8% — model implies +0.7pp"
6. "Causal disclaimer: This is association, not causation"

### Talking Points for Q&A
- **Why HGB over Ridge?** Better validation MAE, captures non-linearities
- **Why negative R²?** Macroeconomic growth is noisy; baselines are strong
- **Why 14 features?** Coverage filter on training data only
- **How often retrain?** Annually, or when WDI updates
- **Can I use this for policy?** No — screening only, causal research needed

---

## Appendix Slides (if time permits)

### A1: Data Pipeline Details
### A2: Hyperparameter Tuning Results
### A3: Bootstrap Confidence Intervals
### A4: Worst Error Analysis
### A5: Country-Level Metrics