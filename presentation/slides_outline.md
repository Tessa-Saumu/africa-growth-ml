# Africa Growth Explorer — Presentation Outline

**Format:** 5–10 minute presentation / demo video
**Scripted runtime: 9 minutes 15 seconds**, leaving 45s of buffer inside the 10-minute ceiling.

> Every number on slides 3, 5 and 6 is pasted from `reports/generated/`, which
> `scripts/build_report_assets.py` builds from the saved model files.
> Do not retype them by hand. `tests/test_report_assets.py` fails the build if
> any document quotes a number the saved model does not support.

---

## Required-section coverage

The brief names six items. Each maps to a numbered slide:

| Required item | Slide | Time |
|---|---|---:|
| Problem Statement | 2 | 45s |
| Dataset Used | 3 | 50s |
| Analysis Process | 4 | 70s |
| Model Performance | 5 | 85s |
| Key Insights | 6 | 65s |
| Recommendations | 8 | 55s |
| *(supporting: title, demo, rigor, close)* | 1, 7, 9, 10 | 175s |

---

## Slide 1: Title (20 seconds)

**Africa Growth Explorer**
*Predicting next-year GDP per capita growth across African economies from World Bank development indicators*

**Question:** can recent development indicators predict near-term growth?

**Answer:** no. The model does no better than guessing the historical average. The contribution is a build process careful enough to establish that rather than hide it.

**Presenter:** [Your Name] · **Date:** [Date] · **Program:** AnalystLab Data Science Internship Capstone

> **Say:** "I'll give you the conclusion first, because it shapes everything after it. The model does no better than guessing the average — and I can show you why that's the correct answer rather than a failed one."

---

## Slide 2: Problem Statement (45 seconds)

### The task
Use this year's development indicators to predict next year's GDP per capita growth, across the 54 African UN member states.

### Why it's worth doing carefully
Development analysts need screening tools. But a growth prediction is only useful if you can state how much better it is than "predict the average" — and that comparison is the step most portfolio projects skip.

### Framing
- **Not:** a tool that tells you what causes growth
- **Is:** a tool that finds patterns, plus a build process that refuses to ship a model which fails to beat the obvious alternatives
- **Users:** analysts, researchers, policy teams, NGOs

### The bar, set before any modelling
Beat all three trivial rules — predict the average, predict this year's growth, predict each country's own average — on the tuning data. Then check on untouched data whether the winning margin is big enough to be real. A model that cannot clear that bar does not ship.

---

## Slide 3: Dataset Used (50 seconds)

### Source
World Bank **World Development Indicators** (WDI), 2000–2024. Public, no licence restriction, cited in `data/README.md`.

### Shape
- 14 indicators across 6 themes — 13 report often enough to keep, plus the current-year growth column
- **52 countries** in the data as committed; the configuration lists all 54. Mauritius and Sudan are missing from this vintage and arrive with the next refresh. Stated here rather than rounded up to "54"
- 1,300 country-year rows; 1,205 usable once each row is paired with the following year's growth

### What the model predicts
```
Indicators at year t  →  Growth at year t+1
Ghana 2019 indicators  →  Ghana 2020 GDP per capita growth
```

### How the years were divided — note the target years run one later
| Portion | Input years | Growth years predicted | n |
|---|---|---|---|
| Training | 2000–2017 | 2001–2018 | 905 |
| Tuning | 2018–2020 | **2019–2021** (includes the COVID crash) | 150 |
| Testing | 2021–2023 | **2022–2024** (untouched, opened once) | 150 |

> **Say:** "The years being predicted matter more than the input years. Tuning covers the COVID crash; testing covers the recovery. That mismatch forces a design decision on the next slide."

---

## Slide 4: Analysis Process (70 seconds)

### The build — `scripts/finalize_model.py`, run in this fixed order
```
Raw WDI download
→ select African countries by explicit country code
→ build the country-year table, resolve duplicates
→ attach each row's following-year growth
→ drop indicators reporting under 60% of the time,
     measured on TRAINING rows only
→ tune settings using training years only, by fitting on an early
     stretch and checking the years just after, then widening:
     trees  depth {2,3} × learning rate {0.01,0.03,0.05} × rounds {100,200}
     Ridge  penalty strength {1 … 3000}
→ score the best of each family on the TUNING data
→ CHECK AGAINST THE TRIVIAL RULES — fail ⇒ nothing is saved, build exits
→ retrain on training data only, as decided in advance
→ OPEN THE TEST SET, exactly once: score, and test whether the margin is real
→ stamp every saved file with a data fingerprint, git commit, library versions
```

### What the data showed, and how it changed the model
- Development levels — electricity, internet, urbanisation, life expectancy, GDP per capita — move together, correlating at **0.45–0.71**. They are five measurements of roughly the same underlying thing, not five independent signals
- Current-year growth barely moves with any of them (correlations of 0.13 or less)
- Missing data is not random — it clusters in particular countries, so the "does this indicator report often enough?" test runs on training rows only

### Models compared
1. **Ridge regression** — a straight-line model whose coefficients carry direction
2. **Gradient-boosted trees** — deployed; won on tuning data and beat the trivial rules

### The three trivial rules
Predict the average · predict this year's growth · predict each country's own past average (using only years already seen)

> **Say:** "I decided how to retrain before opening the test set. The tuning years contain the COVID crash and the test years are the recovery — train across that break and predictions come out systematically low. I locked in training-only beforehand, and I report the alternative so you can see the cost."

---

## Slide 5: Model Performance (85 seconds)

### Test set — growth years 2022–2024, n=150

Lower MAE and RMSE are better; both are average error sizes in percentage points.

| Model | MAE | RMSE | R² | Direction right | Always-"up" scores | Gain |
|---|---:|---:|---:|---:|---:|---:|
| Predict the average | 1.90 | 2.84 | −0.00 | 80.7% | 80.7% | 0.0 pts |
| Predict this year's growth | 2.23 | 4.52 | −1.54 | 77.3% | 80.7% | −3.3 pts |
| Predict country's own average | 1.94 | 2.88 | −0.03 | 78.0% | 80.7% | −2.7 pts |
| **Gradient-boosted trees (deployed)** | **1.82** | **2.79** | **0.03** | 80.7% | 80.7% | 0.0 pts |

*(The "predict this year's growth" rule needs a previous year to exist, so it is scored only on rows where all models can compete — report §7.)*

### The number that decides the story
> **The model beats the predict-the-average rule by 0.07 percentage points.**
> **Resample the 150-row test set 5,000 times and that margin runs from 0.04 worse to 0.19 better.**
> **A range that includes losing means the lead is not real. The two perform the same. That is the finding.**

> **Say:** "The model wins by seven hundredths of a point. But there are only 150 rows in the test set, so I resampled it five thousand times to see how stable that is. On a good number of those resamples the model actually loses. When your margin includes losing, you don't have a margin — you have noise."

### The tuning data, where the model was actually chosen
Ridge 4.00 · trees **3.89** · predict-the-average 4.04 — the trees cleared the bar by 3.7%, a margin well inside noise at 150 rows. The same non-result shows up here.

### Reading the direction figure honestly
The model calls the direction of growth right 80.7% of the time. But growth was positive in 80.7% of test years — so always saying "up", using no data at all, scores exactly the same. The gap between the model and that trivial rule is **zero**. Scoring up-years and down-years separately and averaging gives **51.3%**, a coin flip.

> **Say:** "If I put 80.7% on a slide with no context, it looks like the strongest number in the deck. It's the weakest. It's just how often growth happened to be positive."

> **Say:** "R² is +0.03. That is not a model that explains growth — it's a model that has correctly learned to predict the mean and stop."

---

## Slide 6: Key Insights (65 seconds)

### 1. Only 2 of 14 indicators made a reliable difference
Shuffle one indicator's column at random, re-score, and see how much worse the model gets:
- GDP per capita **0.046** [0.018, 0.085]
- Population growth **0.017** [0.000, 0.037]
- For the other twelve — including electricity and inflation — the range of plausible values includes zero, meaning the model was not really using them

Scale check: scrambling the strongest indicator worsens predictions by 0.046, against typical errors near 3.9 — roughly one percent. Real enough to measure, far too small to act on.

### 2. This measures size, never direction
There is no "positive driver / negative driver" column, because shuffling a column tells you how much the model leaned on an indicator, not which way it pushed. An earlier draft had that column; it was a category error and was removed.

### 3. Even the straight-line structure is faint
Ridge settled on the strongest penalty in the grid — 3000, the largest value offered. Pulling predictions as hard as possible toward the average is the best a straight-line model can do here. Every coefficient is 0.14 or smaller.

### 4. The big misses are events, not model failure
Every worst case is a shock year: Libya 2021 (13.15 pp off), Cabo Verde 2021 (12.53 pp), Equatorial Guinea 2022 (11.01 pp). Conflict, tourism collapse, oil. No set of annual indicators anticipates these.

### 5. Why believe this result
The same dataset previously produced a "working" model that was in fact worse than a constant — its test error was nearly double that of simply predicting the average — because the winner had been picked using the test set, and the report's statistics were invented. Fixing the process removed the mirage. This is what the honest machinery reports.

---

## Slide 7: Live Demo — Streamlit App (110 seconds)

**Page 1 — Overview:** the problem, the data, the model, the headline result with its caveat stated up front, and the reminder that none of this shows cause and effect.

**Page 2 — Explore Africa:** country selector, growth over time (what happened vs what the model was asked to predict), indicator trends, regional comparison.

**Page 3 — Model Performance:** the "this margin is not real" banner; how the model scores against the trivial rules on both test and tuning data, always showing what always-saying-"up" would score; where predictions land against reality; the errors; which indicators the model actually used, with the ones it ignored listed separately; the Ridge direction panel; results year by year.

**Page 4 — Scenario Explorer** ← the decision-support feature
1. Pick a country and year → its actual indicator values load, with a notice if any were filled in
2. Sliders stay inside the range the model trained on, and warn past the edges — inflation warns above ~49.5 rather than 92
3. If a country's real value sits outside that range, the slider is pulled to the edge **and says so**
4. The change table re-runs the model once per indicator, and says plainly that the effects do not add up
5. The cause-and-effect warning stays on screen

*Narrate from live output only. No figures are scripted here — an earlier draft's illustrative numbers were removed.*

---

## Slide 8: Recommendations (55 seconds)

### For anyone using this tool
1. **Do not allocate funding based on these predictions.** They cannot be told apart from simply guessing the average. This is the first recommendation because it is the one with consequences.
2. **Use it to compare and describe** development profiles — the Explore page. That is what the data supports.
3. **Treat every scenario change as the model's response**, never as the effect of a policy.

### For the next round of work
4. **Find faster-moving data** — satellite nightlights, port and air-traffic volumes, mobile-money flows, survey expectations — rather than more annual indicators. Tying with the average argues for better signal, not more models.
5. **Keep the automatic check against trivial rules, and keep deciding the rules before looking at the data.** That is the reusable asset from this project.
6. **Re-download the WDI data** to restore Mauritius and Sudan before quoting country coverage externally.
7. **Watch for periods that behave differently.** This result is partly a statement about a 25-year window containing three very different economic eras.

---

## Slide 9: Engineering & Rigor (40 seconds)

- **Nothing from the future leaks backwards:** the coverage test runs on training rows only; filling gaps and rescaling happen inside the model, refitted per fold; each row's target comes from the following year within the same country; the test set stays closed until one final scoring pass
- **A model worse than guessing cannot ship** — the check runs before anything is saved, and a failure exits with an error and writes nothing
- **Settings genuinely tuned on training years only**, fitting on an early stretch and checking the years just after, with the fold-by-fold results committed (`models/cv_results_*.csv`)
- **The retraining policy was fixed in advance**, and the alternative is reported as a cost: error 2.03, predictions 0.86 points low
- **The headline comparison was tested for significance**, and reported whatever it said
- **Every saved file records its origin:** data fingerprint, git commit, library versions, which years went where. Two runs produce identical numbers
- **86 passing tests**, including deliberate sabotage: corrupt the test answers and model selection stays byte-for-byte identical; force the trivial-rule check to fail and confirm nothing gets saved
- **Notebooks run start to finish**, with no hardcoded paths, loading the shipped model rather than picking a new one

---

## Slide 10: Close (15 seconds)

1. **A result you can defend beats a win you can't.** The process is the contribution.
2. **Make your build check the obvious alternatives** — "does this beat what it replaces?" belongs in the code, not the discussion section.
3. **Generate every number in every document from the saved model**, so the write-up cannot drift from what was actually built.

- **GitHub:** this repository — code, saved model, tests, executed notebooks, full report
- **App:** verified locally via `streamlit run app.py`; no public URL in this submission (README → Deployment)
- **Report:** `reports/capstone_report.md`
- **Contact:** [your email]

---

## Q&A talking points *(not counted in runtime)*

- **Why trees over Ridge?** On the tuning data the best tree model missed by 3.89 points against Ridge's 4.00, and it beat the trivial rules.
- **Is tying with the trivial rule a failure?** No. It is a measured, tested statement about what this data can and cannot predict, produced by a process with no way to fake a result.
- **Why did early stopping matter?** The previous version overfit because scikit-learn's automatic setting switches itself off below 10,000 samples, and there are 905 training rows. Setting it explicitly bounds the model — it now stops at 45 rounds out of 200.
- **Why fix the retraining policy in advance?** The tuning years contain the COVID crash. Training across it pushes predictions systematically low: error 2.03 and 0.86 points low, against 1.82 and 0.08 high for training-only.
- **Why not tune harder?** Hunting for settings that score well on the test set is exactly how the previous version shipped a bad model. The grid is small, fixed in advance, and scored only within training years.
- **How often retrain?** Annually, as WDI updates. The trivial-rule check and the significance test must pass each time.
- **Can this inform policy?** As background screening only. Cause-and-effect claims need experiments or natural experiments.
- **What would change your mind?** Faster-moving data, or a test window that isn't a pandemic recovery. Both are in Recommendations.

---

## Appendix *(only if asked)*

- A1: Fold-by-fold tuning results (`models/cv_results_*.csv`)
- A2: How the ranges around each number were computed, and the head-to-head significance test (report §7)
- A3: The largest misses, every row checked against the frozen predictions (report §7)
- A4: Which indicators were kept and why — secondary school enrolment was dropped for reporting only 59.9% of the time
