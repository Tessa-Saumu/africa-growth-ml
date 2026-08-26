# Africa Growth Explorer - Implementation Plan (v2 - All Issues Fixed)

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an end-to-end ML decision-support system that predicts near-term GDP per capita growth for African countries using World Bank WDI data, deployed as a Streamlit Cloud application.

**Architecture:** Modular `src/` pipeline (data → features → train → evaluate → visualize) with serialized sklearn pipelines. Streamlit app loads pre-trained artifacts for inference. Temporal validation with chronological train/val/test splits prevents data leakage. Single source of truth: `model_metadata.json` defines the model's feature contract; app reads from metadata, not config.

**Tech Stack:** Python 3.11, pandas, numpy, scikit-learn (HistGradientBoostingRegressor, Ridge), matplotlib, seaborn, streamlit, joblib, pyyaml, pyarrow

---

## Change Log

| Version | Date | Changes |
|---------|------|---------|
| v1 | 2026-08-26 | Initial plan |
| v2 | 2026-08-26 | Fixed 6 blockers (B1-B6), 7 gaps, 12 minors. Added report, presentation, data audit, metadata writing tasks. |

---

## AGENTS.md Compliance

Every task in this plan must adhere to these rules:

| Rule | How Enforced |
|------|-------------|
| 1. Verify code | Every task ends with `pytest` run + reported result |
| 2. Tests for every src/ file | Every `src/<module>.py` has `tests/test_<module>.py` |
| 3. Docstrings + type hints | Module docstring at top of every file; function docstrings with Args/Returns; type hints on all signatures |
| 4. Logging, no print() | `logging` module used throughout; zero `print()` statements |
| 5. No silent failures | All error paths raise exceptions or log warnings |
| 6. No leakage | Temporal split enforced; imputer fitted only on training data inside sklearn pipeline; log transform inside pipeline via FunctionTransformer |
| 7. Lean notebooks | Notebooks: Markdown → code → output → interpretation; logic lives in `src/` |
| 8. Deliberate visuals | `src/visualization.py` defines project palette; all charts use consistent styling |
| 9. Minimal dependencies | Only essential packages in requirements.txt; xgboost not needed |
| 10. Follow spec | Streamlit only, no FastAPI/Flask/Render |
| 11. Smallest change | Inspect existing code first; minimal change; update tests |
| 12. Final status | Every response states: what changed, tests run, result, VERIFIED/NOT VERIFIED |

---

## Skills That Assist (No Conflicts)

| Skill | Relevance | How It Helps |
|-------|-----------|-------------|
| `scikit-learn` | HIGH | Pipeline construction, Ridge/GradientBoosting patterns, evaluation metrics |
| `ml-pipeline-workflow` | HIGH | End-to-end pipeline architecture |
| `python-testing-patterns` | HIGH | pytest fixtures, TDD for every module |
| `python-observability` | HIGH | `logging` usage throughout |
| `writing-plans` | HIGH | Plan structure and task granularity |

None of these introduce FastAPI, Flask, or alternative deployment strategies. They all align with the Streamlit-only spec.

---

## File Structure Map

```
africa-growth-ml/
├── app.py                          # Streamlit application (reads metadata for feature contract)
├── requirements.txt                # Dependencies (includes pytest)
├── requirements-dev.txt            # Dev-only dependencies (optional)
├── README.md                       # Project documentation
├── LICENSE                         # MIT License
├── .gitignore                      # Ignores data/raw/ and logs/ only
├── config/
│   └── indicators.yaml             # Feature candidates, model settings, year ranges
├── data/
│   ├── README.md                   # Data source instructions
│   └── processed/
│       ├── model_data.parquet      # Final modeling dataset (committed)
│       └── country_metadata.csv    # Country reference table (committed)
├── models/
│   ├── growth_model.joblib         # Serialized sklearn pipeline (committed)
│   └── model_metadata.json         # Feature contract, metrics, split years (committed)
├── notebooks/
│   ├── 01_data_profiling.ipynb     # EDA + data audit
│   └── 02_model_evaluation.ipynb   # Model comparison + error analysis + bootstrap CI
├── src/
│   ├── __init__.py
│   ├── config.py                   # Configuration loading, constants
│   ├── data.py                     # Data loading, filtering, reshaping, __main__ runner
│   ├── features.py                 # Panel construction, target creation, selection, __main__ runner
│   ├── train.py                    # Baselines, pipelines, training, serialization, metadata writing
│   ├── evaluate.py                 # Metrics, error analysis, importance
│   └── visualization.py            # Chart functions with project visual language
├── tests/
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_data.py
│   ├── test_features.py
│   ├── test_train.py
│   ├── test_evaluate.py
│   └── test_visualization.py
├── reports/
│   └── capstone_report.pdf         # Final report
├── presentation/
│   └── slides_outline.md           # Slide content (user builds visuals separately)
└── logs/                           # Local logging (gitignored)
```

---

## Execution Phases

### Phase 0: Scaffolding (Tasks 0.1–0.7)
### Phase 1: Data Pipeline (Tasks 1.1–1.4)
### Phase 2: Modeling Pipeline (Tasks 2.1–2.4)
### Phase 3: Application & Deployment (Tasks 3.1–3.5)
### Phase 4: Deliverables (Tasks 4.1–4.2)

---

# Phase 0: Project Scaffolding & Configuration

> **Goal:** Establish project structure, configuration system, dependency management, and data audit.

---

## Task 0.1: Populate requirements.txt

**Files:** Modify: `requirements.txt`

- [ ] **Step 1: Write requirements.txt**

```txt
pandas>=2.0,<3.0
numpy>=1.24,<2.0
scikit-learn>=1.3,<2.0
matplotlib>=3.7,<4.0
seaborn>=0.12,<1.0
streamlit>=1.30,<2.0
joblib>=1.3,<2.0
pyarrow>=14.0,<15.0
pyyaml>=6.0,<7.0
pytest>=7.0
```

Note: `pytest` is included here because every task requires running tests, and Streamlit Cloud/CI reproducibility matters. Plotly is removed - not needed for this project's charts.

- [ ] **Step 2: Verify installation**

Run: `pip install -r requirements.txt`
Expected: All packages install without errors

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "feat: add project dependencies including pytest"
```

---

## Task 0.2: Create .gitignore (B1 FIX)

**Files:** Modify: `.gitignore`

**B1 FIX:** `models/` and `data/processed/` are NOT ignored. Streamlit Cloud pulls from GitHub and needs these files. Only `data/raw/` (30MB CSV) and `logs/` are ignored.

- [ ] **Step 1: Write .gitignore**

```gitignore
# Raw data (too large for git - ~30MB CSV)
data/raw/
*.zip

# Logs (local only)
logs/

# Python
__pycache__/
*.pyc
*.pyo
.venv/
venv/
env/

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/

# Environment
.env
```

Note: `models/` and `data/processed/` are intentionally NOT ignored. They are small (joblib is ~hundreds of KB, parquet is a few MB) and must be deployed to Streamlit Cloud.

- [ ] **Step 2: Commit**

```bash
git add .gitignore
git commit -m "chore: add .gitignore - models and processed data are committed"
```

---

## Task 0.3: Create LICENSE

**Files:** Create: `LICENSE`

- [ ] **Step 1: Write MIT License**

```text
MIT License

Copyright (c) 2026 Africa Growth Explorer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

- [ ] **Step 2: Commit**

```bash
git add LICENSE
git commit -m "chore: add MIT license"
```

---

## Task 0.4: Create config/indicators.yaml

**Files:** Create: `config/indicators.yaml`, Create: `tests/__init__.py`, Create: `tests/test_config.py`

- [ ] **Step 1: Write config/indicators.yaml**

```yaml
# Africa Growth Explorer - Indicator Configuration
# Feature CANDIDATES. Final feature list is determined by coverage analysis
# and written to model_metadata.json at training time.

target:
  code: "NY.GDP.PCAP.KD.ZG"
  name: "GDP per capita growth (annual %)"
  prediction_horizon_years: 1

features:
  - code: "NY.GDP.PCAP.CD"
    name: "GDP per capita (current US$)"
    theme: "Economic level"
  - code: "EG.ELC.ACCS.ZS"
    name: "Access to electricity (% of population)"
    theme: "Infrastructure"
  - code: "IT.NET.USER.ZS"
    name: "Individuals using the Internet (% of population)"
    theme: "Technology"
  - code: "NE.GDI.TOTL.ZS"
    name: "Gross capital formation (% of GDP)"
    theme: "Investment"
  - code: "BX.KLT.DINV.WD.GD.ZS"
    name: "Foreign direct investment, net inflows (% of GDP)"
    theme: "External investment"
  - code: "NE.TRD.GNFS.ZS"
    name: "Trade (% of GDP)"
    theme: "Openness"
  - code: "FP.CPI.TOTL.ZG"
    name: "Inflation, consumer prices (annual %)"
    theme: "Macroeconomic stability"
  - code: "SL.UEM.TOTL.ZS"
    name: "Unemployment, total (% of total labor force)"
    theme: "Labour market"
  - code: "SP.DYN.LE00.IN"
    name: "Life expectancy at birth, total (years)"
    theme: "Health"
  - code: "SE.SEC.ENRR"
    name: "School enrollment, secondary (% gross)"
    theme: "Education"
  - code: "FS.AST.PRVT.GD.ZS"
    name: "Domestic credit to private sector (% of GDP)"
    theme: "Financial development"
  - code: "NE.CON.GOVT.ZS"
    name: "General government final consumption expenditure (% of GDP)"
    theme: "Public sector"
  - code: "SP.URB.TOTL.IN.ZS"
    name: "Urban population (% of total population)"
    theme: "Demographics"
  - code: "SP.POP.GROW"
    name: "Population growth (annual %)"
    theme: "Demographics"

# Log-transform candidates: features that benefit from log1p scaling.
# These are applied INSIDE the sklearn pipeline via FunctionTransformer,
# so the app never sees _log columns - it feeds raw WDI values.
log_transform_candidates:
  - "NY.GDP.PCAP.CD"

geographic:
  # "Africa" matches all African regions in WDI metadata (North + Sub-Saharan)
  region_filter: "Africa"

temporal:
  min_year: 2000
  train_end: 2017
  val_end: 2020
  # COVID note: 2020 shock sits in validation (informs model selection).
  # Test is post-COVID 2021+. This is a deliberate decision documented in the report.

model:
  random_state: 42
  ridge_alpha: 1.0
  hgb_max_iter: 1000
  hgb_learning_rate: 0.05
  hgb_max_depth: 5
```

- [ ] **Step 2: Write tests/test_config.py (B6 FIX - syntax corrected)**

```python
"""Tests for configuration loading and validation."""
from pathlib import Path
import pytest
from src.config import load_config, Config


def test_load_config_returns_config_instance():
    """load_config should return a Config dataclass."""
    config = load_config(Path("config/indicators.yaml"))
    assert isinstance(config, Config)


def test_config_has_required_keys():
    """Config should contain features, target, and country settings."""
    config = load_config(Path("config/indicators.yaml"))
    assert len(config.features) > 0
    assert config.target_code is not None
    assert config.random_state == 42  # B6 FIX: was "config RANDOM_STATE" (syntax error)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with `ImportError` or `ModuleNotFoundError`

- [ ] **Step 4: Write src/__init__.py**

```python
"""Africa Growth Explorer - ML Decision Support System."""
```

- [ ] **Step 5: Write src/config.py**

```python
"""Configuration loading and validation for the Africa Growth Explorer.

Loads indicator definitions, model hyperparameters, and temporal split
boundaries from a YAML configuration file.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import yaml
import logging

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for a single WDI feature indicator."""
    code: str
    name: str
    theme: str


@dataclass
class Config:
    """Top-level project configuration loaded from indicators.yaml."""
    features: List[FeatureConfig]
    target_code: str
    target_name: str
    prediction_horizon_years: int
    region_filter: str
    min_year: int
    train_end: int
    val_end: int
    log_transform_candidates: List[str]
    random_state: int = 42
    ridge_alpha: float = 1.0
    hgb_max_iter: int = 1000
    hgb_learning_rate: float = 0.05
    hgb_max_depth: int = 5


def load_config(path: Path = Path("config/indicators.yaml")) -> Config:
    """Load project configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Config dataclass with all project settings.
    """
    logger.info("Loading configuration from %s", path)
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    features = [
        FeatureConfig(
            code=feat["code"],
            name=feat["name"],
            theme=feat.get("theme", ""),
        )
        for feat in raw["features"]
    ]

    config = Config(
        features=features,
        target_code=raw["target"]["code"],
        target_name=raw["target"]["name"],
        prediction_horizon_years=raw["target"]["prediction_horizon_years"],
        region_filter=raw["geographic"]["region_filter"],
        min_year=raw["temporal"]["min_year"],
        train_end=raw["temporal"]["train_end"],
        val_end=raw["temporal"]["val_end"],
        log_transform_candidates=raw.get("log_transform_candidates", []),
        random_state=raw["model"]["random_state"],
        ridge_alpha=raw["model"]["ridge_alpha"],
        hgb_max_iter=raw["model"]["hgb_max_iter"],
        hgb_learning_rate=raw["model"]["hgb_learning_rate"],
        hgb_max_depth=raw["model"]["hgb_max_depth"],
    )
    logger.info(
        "Loaded %d features, target=%s, region=%s",
        len(config.features), config.target_code, config.region_filter
    )
    return config
```

- [ ] **Step 6: Run tests**

Run: `pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add config/indicators.yaml src/config.py src/__init__.py tests/__init__.py tests/test_config.py
git commit -m "feat: add project configuration and config loader"
```

---

## Task 0.5: Create data directories and data/README.md

**Files:** Create: `data/processed/`, Create: `data/README.md`

- [ ] **Step 1: Create directories**

```bash
mkdir -p data/processed
mkdir -p models
mkdir -p logs
```

- [ ] **Step 2: Write data/README.md (fixed: no contradiction about tracked files)**

```markdown
# Data Source

## World Bank World Development Indicators (WDI)

**URL:** https://datatopics.worldbank.org/world-development-indicators/

## Download Instructions

1. Visit the WDI data page above
2. Click "Download" and select the CSV format
3. Save `WDI_CSV.zip` to this directory
4. Unzip and identify the data file and metadata file (names may vary - see Data Audit task)

## Expected Files

```
data/
├── raw/
│   └── WDI_Data.csv       # Raw WDI data (NOT committed to git - too large)
├── processed/
│   ├── model_data.parquet  # Final modeling dataset (committed to git)
│   └── country_metadata.csv # Country reference table (committed to git)
└── README.md
```

## Running the Pipeline

```bash
# Step 1: Load and filter raw data (unzip WDI_CSV.zip first)
python -m src.data

# Step 2: Build country-year panel
python -m src.features

# Or run the notebooks:
jupyter notebook notebooks/01_data_profiling.ipynb
```

## Data Notes

- Raw WDI data is ~30MB and is NOT committed to git
- Processed outputs in `data/processed/` ARE committed (they are small)
- Model artifacts in `models/` ARE committed (they are small)
- The pipeline logs all dimensions and coverage statistics
```

- [ ] **Step 3: Create .gitkeep for data/processed/**

```bash
touch data/processed/.gitkeep
```

- [ ] **Step 4: Commit**

```bash
git add data/README.md data/processed/.gitkeep models/ logs/
git commit -m "chore: add data directories and README"
```

---

## Task 0.6: Data Audit (MINOR FIX - verify actual file names)

**Files:** No persistent file changes. This is a one-time audit.

**Why this task exists:** The WDI CSV zip's actual contents vary between downloads. `WDICountry.csv` vs `WDI_Country.csv`, folder nesting, and the metadata region column name must be confirmed before the data pipeline runs.

- [ ] **Step 1: Unzip and list files**

```bash
cd data/raw
unzip -l WDI_CSV.zip
```

Record:
- Exact data file name (e.g., `WDICountry.csv`, `WDI_Data.csv`)
- Exact metadata file name (e.g., `WDICountry.csv`, `WDI_Country.csv`)
- Whether there is folder nesting (e.g., `WDI_CSV/WDICountry.csv`)

- [ ] **Step 2: Check metadata columns**

```python
import pandas as pd
# Replace with actual file name from step 1
meta = pd.read_csv("data/raw/WDICountry.csv", encoding="utf-8")
print(meta.columns.tolist())
# Look for: CountryCode, Region, IncomeGroup
print(meta[["CountryCode", "Region"]].head(20))
```

Record:
- Exact column name for country code (likely `CountryCode`)
- Exact column name for region (likely `Region`)
- Region values that contain "Africa" (e.g., "Sub-Saharan Africa", "Africa Eastern and Southern", "Africa Western and Central")

- [ ] **Step 3: Update constants in src/data.py**

Update `WDI_DATA_FILE` and `WDI_METADATA_FILE` constants with actual file names. Update `filter_african_countries` to use the actual region column name and values.

- [ ] **Step 4: Commit**

```bash
git add src/data.py  # Only if constants changed
git commit -m "chore: update WDI file names from data audit"
```

---

## Task 0.7: Write src/__main__.py runners (GAP FIX)

**Files:** Modify: `src/data.py`, Modify: `src/features.py`

**GAP FIX:** `python -m src.data` and `python -m src.features` currently don't work because there are no `__main__` blocks. The README references these commands.

- [ ] **Step 1: Add __main__ block to src/data.py**

Append to end of `src/data.py`:

```python
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raw_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw/WDI_Data.csv")
    metadata_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/raw/WDICountry.csv")
    try:
        df = load_wdi_csv(raw_path)
        metadata = load_wdi_metadata(metadata_path) if metadata_path.exists() else None
        african_df = filter_african_countries(df, metadata=metadata)
        indicator_codes = [f.code for f in load_config().features]
        indicator_codes.append(load_config().target_code)
        filtered = filter_indicators(african_df, indicator_codes)
        year_cols = [str(y) for y in range(load_config().min_year, 2025)]
        long = reshape_wide_to_long(filtered, year_cols)
        panel = pivot_to_country_year(long)
        numeric_cols = [c for c in panel.columns if c not in ["iso3", "country_name", "year"]]
        panel = clean_numeric(panel, numeric_cols)
        log_dataset_info(panel, "Final Panel")
        panel.to_parquet("data/processed/model_data.parquet", index=False)
        logger.info("Saved to data/processed/model_data.parquet")
    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        sys.exit(1)
```

- [ ] **Step 2: Add __main__ block to src/features.py**

Append to end of `src/features.py`:

```python
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    panel_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/processed/model_data.parquet")
    try:
        panel = pd.read_parquet(panel_path)
        config = load_config()
        panel = create_target(panel, config.target_code)
        feature_cols = [c for c in panel.columns if c not in
                       ["iso3", "country_name", "year", "target_next_year"]]
        panel = select_features_by_coverage(panel, feature_cols, min_coverage=0.6)
        panel.to_parquet(panel_path, index=False)
        logger.info("Updated panel saved to %s", panel_path)
    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        sys.exit(1)
```

- [ ] **Step 3: Commit**

```bash
git add src/data.py src/features.py
git commit -m "feat: add __main__ runners for python -m execution"
```

---

## Phase 0 Verification

- [ ] `pip install -r requirements.txt` succeeds
- [ ] `pytest tests/test_config.py -v` passes
- [ ] `config/indicators.yaml` loads correctly
- [ ] `.gitignore` does NOT ignore `models/` or `data/processed/`
- [ ] All directories exist: `data/processed/`, `models/`, `logs/`
- [ ] Data audit completed: file names and column names confirmed

---

# Phase 1: Data Pipeline

> **Goal:** Load WDI data, profile it, clean it, and build the country-year panel.

---

## Task 1.1: Write src/data.py - Data Loading & Filtering

**Files:** Create: `src/data.py`, Create: `tests/test_data.py`

- [ ] **Step 1: Write test_data.py (B6 FIX - explicit_codes provided)**

```python
"""Tests for data loading, filtering, and reshaping."""
import pandas as pd
import numpy as np
import pytest
from pathlib import Path
from src.data import (
    load_wdi_csv,
    filter_african_countries,
    filter_indicators,
    reshape_wide_to_long,
    pivot_to_country_year,
    clean_numeric,
    log_dataset_info,
)


@pytest.fixture
def sample_wdi_df():
    """Create a small sample WDI dataframe for testing."""
    data = {
        "Country Name": ["Ghana", "Kenya", "Nigeria", "Sub-Saharan Africa"],
        "Country Code": ["GHA", "KEN", "NGA", "SSF"],
        "Indicator Name": ["GDP per capita growth", "GDP per capita growth"],
        "Indicator Code": ["NY.GDP.PCAP.KD.ZG", "NY.GDP.PCAP.KD.ZG"],
        "2018": [6.2, 5.1, 2.0],
        "2019": [7.1, 5.4, 2.2],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_metadata():
    """Create a small sample metadata dataframe."""
    return pd.DataFrame({
        "CountryCode": ["GHA", "KEN", "NGA", "SSF"],
        "Region": ["Sub-Saharan Africa", "Sub-Saharan Africa", "Sub-Saharan Africa", "Sub-Saharan Africa"],
    })


def test_filter_african_countries_with_metadata(sample_wdi_df, sample_metadata):
    """Using metadata, aggregates should be filtered out."""
    result = filter_african_countries(sample_wdi_df, metadata=sample_metadata)
    assert "Sub-Saharan Africa" not in result["Country Name"].values
    assert len(result) == 3  # GHA, KEN, NGA only


def test_filter_african_countries_with_explicit_codes(sample_wdi_df):
    """B6 FIX: Using explicit_codes (not calling with no args which raises ValueError)."""
    result = filter_african_countries(sample_wdi_df, explicit_codes=["GHA", "KEN", "NGA"])
    assert "Sub-Saharan Africa" not in result["Country Name"].values
    assert len(result) == 3


def test_filter_african_countries_raises_without_args(sample_wdi_df):
    """Should raise ValueError when neither metadata nor explicit_codes provided."""
    with pytest.raises(ValueError, match="Either metadata or explicit_codes"):
        filter_african_countries(sample_wdi_df)


def test_filter_indicators_selects_correct_codes(sample_wdi_df):
    """Only selected indicator codes should remain."""
    codes = ["NY.GDP.PCAP.KD.ZG"]
    result = filter_indicators(sample_wdi_df, codes)
    assert result["Indicator Code"].nunique() == 1


def test_reshape_wide_to_long(sample_wdi_df):
    """Wide year columns should become a 'year' column."""
    result = reshape_wide_to_long(sample_wdi_df, year_columns=["2018", "2019"])
    assert "year" in result.columns
    assert result["year"].dtype in [int, float]
    # 3 countries (GHA, KEN, NGA after filtering) x 1 indicator x 2 years = 6
    assert len(result) == 6


def test_clean_numeric_handles_blank_strings():
    """Blank strings and placeholders should become NaN."""
    df = pd.DataFrame({"value": ["1.5", "", "..", "N/A", "3.2"]})
    result = clean_numeric(df, ["value"])
    assert pd.isna(result["value"].iloc[1])  # blank
    assert pd.isna(result["value"].iloc[2])  # ..
    assert pd.isna(result["value"].iloc[3])  # N/A
    assert result["value"].iloc[0] == pytest.approx(1.5)
    assert result["value"].iloc[4] == pytest.approx(3.2)


def test_log_dataset_info_logs(caplog):
    """log_dataset_info should log dataset dimensions."""
    import logging
    df = pd.DataFrame({"iso3": ["GHA", "KEN"], "year": [2018, 2019], "val": [1.0, 2.0]})
    with caplog.at_level(logging.INFO):
        log_dataset_info(df, "Test")
    assert "Shape" in caplog.text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_data.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write src/data.py (MINOR FIX - utf-8 encoding, region uses config value)**

```python
"""Data loading, filtering, and reshaping for the Africa Growth Explorer.

Loads raw WDI CSV data, filters to African countries and selected indicators,
reshapes from wide to long format, and produces a clean country-year panel.
"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# Actual file names confirmed during data audit (Task 0.6)
WDI_DATA_FILE = "WDI_Data.csv"
WDI_METADATA_FILE = "WDICountry.csv"


def load_wdi_csv(path: Path) -> pd.DataFrame:
    """Load the raw WDI CSV file.

    Args:
        path: Path to WDI CSV file.

    Returns:
        Raw dataframe with all countries, indicators, and years.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If the file is empty.
    """
    if not path.exists():
        raise FileNotFoundError(f"WDI data file not found: {path}")
    logger.info("Loading WDI data from %s", path)
    df = pd.read_csv(path, encoding="utf-8")  # MINOR FIX: utf-8, not latin-1
    if df.empty:
        raise ValueError(f"WDI data file is empty: {path}")
    logger.info("Raw WDI shape: %s", df.shape)
    return df


def load_wdi_metadata(path: Path) -> pd.DataFrame:
    """Load WDI country metadata for region classification.

    Args:
        path: Path to WDI country metadata CSV.

    Returns:
        Metadata dataframe with CountryCode and Region columns.

    Raises:
        FileNotFoundError: If the metadata file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"WDI metadata file not found: {path}")
    logger.info("Loading WDI metadata from %s", path)
    metadata = pd.read_csv(path, encoding="utf-8")  # MINOR FIX: utf-8
    logger.info("Metadata shape: %s", metadata.shape)
    return metadata


def filter_african_countries(
    df: pd.DataFrame,
    metadata: Optional[pd.DataFrame] = None,
    explicit_codes: Optional[List[str]] = None,
    region_filter: str = "Africa",
) -> pd.DataFrame:
    """Filter to African countries only.

    Args:
        df: Raw WDI dataframe.
        metadata: WDI metadata with Region column. If None, use explicit_codes.
        explicit_codes: Fallback list of ISO3 codes if metadata unavailable.
        region_filter: String to match in Region column (from config).

    Returns:
        Filtered dataframe with African countries only.

    Raises:
        ValueError: If neither metadata nor explicit_codes is provided.
    """
    if metadata is not None and "CountryCode" in metadata.columns:
        african_codes = metadata[
            metadata["Region"].str.contains(region_filter, case=False, na=False)
        ]["CountryCode"].unique()
        logger.info("Found %d African countries from metadata (filter='%s')",
                    len(african_codes), region_filter)
    elif explicit_codes is not None:
        african_codes = explicit_codes
        logger.info("Using %d explicit African country codes", len(african_codes))
    else:
        raise ValueError("Either metadata or explicit_codes must be provided")

    # Exclude known aggregates
    aggregate_codes = {"SSF", "AFE", "AFW", "WLD", "INX", "SSA", "EAS", "ECS",
                       "TEA", "TMN", "SAS", "ECS", "LCN", "EMU", "OED", "PSS",
                       "PST", "UMC", "LIC", "MIC", "HPC", "FCS", "INX"}
    african_codes = [c for c in african_codes if c not in aggregate_codes]

    mask = df["Country Code"].isin(african_codes)
    filtered = df[mask].copy()
    logger.info(
        "Filtered to %d rows (%d countries) from %d total rows",
        len(filtered),
        filtered["Country Code"].nunique(),
        len(df),
    )
    return filtered


def filter_indicators(df: pd.DataFrame, indicator_codes: List[str]) -> pd.DataFrame:
    """Filter to selected WDI indicator codes.

    Args:
        df: WDI dataframe.
        indicator_codes: List of WDI indicator codes to retain.

    Returns:
        Filtered dataframe with selected indicators only.
    """
    mask = df["Indicator Code"].isin(indicator_codes)
    filtered = df[mask].copy()
    logger.info(
        "Filtered to %d indicators (%d rows)",
        filtered["Indicator Code"].nunique(),
        len(filtered),
    )
    return filtered


def reshape_wide_to_long(
    df: pd.DataFrame, year_columns: List[str]
) -> pd.DataFrame:
    """Reshape from wide (year columns) to long (year column) format.

    Args:
        df: Wide-format WDI dataframe.
        year_columns: List of year column names (e.g., ['2000', '2001', ...]).

    Returns:
        Long-format dataframe with iso3, country_name, indicator_code, year, value.
    """
    id_cols = ["Country Code", "Country Name", "Indicator Code", "Indicator Name"]
    available_years = [y for y in year_columns if y in df.columns]
    long = df.melt(
        id_vars=id_cols,
        value_vars=available_years,
        var_name="year",
        value_name="value",
    )
    long = long.rename(columns={
        "Country Code": "iso3",
        "Country Name": "country_name",
        "Indicator Code": "indicator_code",
    })
    long["year"] = pd.to_numeric(long["year"], errors="coerce")
    logger.info("Reshaped to long format: %d rows", len(long))
    return long


def pivot_to_country_year(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot indicators into feature columns for each country-year.

    Args:
        df: Long-format dataframe with indicator_code and value columns.

    Returns:
        Wide dataframe with iso3, country_name, year, and one column per indicator.
    """
    pivoted = df.pivot_table(
        index=["iso3", "country_name", "year"],
        columns="indicator_code",
        values="value",
        aggfunc="first",
    ).reset_index()
    pivoted.columns.name = None
    logger.info("Pivoted to country-year panel: %s", pivoted.shape)
    return pivoted


def clean_numeric(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Convert string columns to numeric, handling blanks and placeholders.

    Args:
        df: Input dataframe.
        columns: List of column names to convert to numeric.

    Returns:
        Dataframe with numeric columns and NaN for invalid values.
    """
    df = df.copy()
    placeholder_patterns = ["", "..", "N/A", "n/a", "NA", "null", "NULL"]
    for col in columns:
        if col in df.columns:
            df[col] = df[col].replace(placeholder_patterns, np.nan)
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def log_dataset_info(df: pd.DataFrame, label: str) -> None:
    """Log dataset dimensions and basic statistics.

    Args:
        df: Dataset to log.
        label: Human-readable label for the log message.
    """
    logger.info("=== %s ===", label)
    logger.info("Shape: %s", df.shape)
    if "iso3" in df.columns:
        logger.info("Countries: %d", df["iso3"].nunique())
    if "year" in df.columns:
        logger.info("Year range: %d - %d", int(df["year"].min()), int(df["year"].max()))
    missing_pct = df.isnull().mean().mean() * 100
    logger.info("Overall missingness: %.1f%%", missing_pct)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    from src.config import load_config
    raw_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(f"data/raw/{WDI_DATA_FILE}")
    metadata_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(f"data/raw/{WDI_METADATA_FILE}")
    try:
        df = load_wdi_csv(raw_path)
        metadata = load_wdi_metadata(metadata_path) if metadata_path.exists() else None
        config = load_config()
        african_df = filter_african_countries(df, metadata=metadata,
                                              region_filter=config.region_filter)
        indicator_codes = [f.code for f in config.features]
        indicator_codes.append(config.target_code)
        filtered = filter_indicators(african_df, indicator_codes)
        year_cols = [str(y) for y in range(config.min_year, 2025)]
        long = reshape_wide_to_long(filtered, year_cols)
        panel = pivot_to_country_year(long)
        numeric_cols = [c for c in panel.columns if c not in ["iso3", "country_name", "year"]]
        panel = clean_numeric(panel, numeric_cols)
        log_dataset_info(panel, "Final Panel")
        panel.to_parquet("data/processed/model_data.parquet", index=False)
        logger.info("Saved to data/processed/model_data.parquet")
    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        sys.exit(1)
```

- [ ] **Step 4: Run all tests**

Run: `pytest tests/test_data.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/data.py tests/test_data.py
git commit -m "feat: implement data loading, filtering, and reshaping pipeline"
```

---

## Task 1.2: Write src/features.py - Feature Engineering & Target Creation

**Files:** Create: `src/features.py`, Create: `tests/test_features.py`

- [ ] **Step 1: Write test_features.py (B3 FIX - no apply_log_transform test)**

```python
"""Tests for feature engineering and target creation."""
import pandas as pd
import numpy as np
import pytest
from src.features import (
    create_target,
    select_features_by_coverage,
    build_feature_matrix,
    create_temporal_split,
)


@pytest.fixture
def sample_panel():
    """Create a sample country-year panel for testing."""
    np.random.seed(42)
    countries = ["GHA", "KEN", "NGA"]
    years = list(range(2000, 2022))
    rows = []
    for c in countries:
        for y in years:
            rows.append({
                "iso3": c,
                "country_name": c,
                "year": y,
                "NY.GDP.PCAP.KD.ZG": np.random.uniform(-2, 8),
                "EG.ELC.ACCS.ZS": np.random.uniform(40, 100),
                "FP.CPI.TOTL.ZG": np.random.uniform(0, 20),
            })
    return pd.DataFrame(rows)


def test_create_target_shifts_gdp_growth(sample_panel):
    """Target should be GDP growth shifted forward by 1 year."""
    result = create_target(sample_panel, "NY.GDP.PCAP.KD.ZG")
    assert "target_next_year" in result.columns
    last_year_mask = result["year"] == result["year"].max()
    assert result.loc[last_year_mask, "target_next_year"].isna().all()
    non_last_mask = result["year"] < result["year"].max()
    assert result.loc[non_last_mask, "target_next_year"].notna().all()


def test_select_features_by_coverage_drops_sparse(sample_panel):
    """Features with low coverage should be dropped."""
    sparse_col = np.where(
        np.random.random(len(sample_panel)) < 0.1, np.nan, 1.0
    )
    sample_panel["SPARSE_FEATURE"] = sparse_col
    result = select_features_by_coverage(
        sample_panel,
        feature_cols=["NY.GDP.PCAP.KD.ZG", "EG.ELC.ACCS.ZS", "SPARSE_FEATURE"],
        min_coverage=0.6,
    )
    assert "SPARSE_FEATURE" not in result.columns
    assert "NY.GDP.PCAP.KD.ZG" in result.columns


def test_select_features_by_coverage_uses_train_only(sample_panel):
    """MINOR FIX: Coverage should be computed on training rows only."""
    # Make all values in train have coverage but test has NaNs
    sample_panel.loc[sample_panel["year"] > 2015, "EG.ELC.ACCS.ZS"] = np.nan
    # If computed on full panel, EG.ELC.ACCS.ZS would be dropped
    # If computed on train only (<=2015), it should be kept
    result = select_features_by_coverage(
        sample_panel,
        feature_cols=["NY.GDP.PCAP.KD.ZG", "EG.ELC.ACCS.ZS"],
        min_coverage=0.6,
        train_mask=sample_panel["year"] <= 2015,
    )
    assert "EG.ELC.ACCS.ZS" in result.columns


def test_create_temporal_split_respects_years(sample_panel):
    """Temporal split should separate by year boundaries."""
    sample_panel = create_target(sample_panel, "NY.GDP.PCAP.KD.ZG")
    train, val, test = create_temporal_split(
        sample_panel, train_end=2015, val_end=2018
    )
    assert train["year"].max() <= 2015
    assert val["year"].min() >= 2016
    assert val["year"].max() <= 2018
    assert test["year"].min() >= 2019


def test_build_feature_matrix_drops_nan_target(sample_panel):
    """Rows with NaN target should be dropped."""
    sample_panel = create_target(sample_panel, "NY.GDP.PCAP.KD.ZG")
    X, y = build_feature_matrix(
        sample_panel,
        ["NY.GDP.PCAP.KD.ZG", "EG.ELC.ACCS.ZS"],
    )
    assert y.notna().all()
    assert len(X) == len(y)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_features.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write src/features.py (B3 FIX - no apply_log_transform, coverage on train only)**

```python
"""Feature engineering, target creation, and temporal splitting.

Builds the country-year panel, creates the next-year GDP growth target,
applies feature selection by coverage (computed on training data only),
and produces train/val/test splits.
"""
import pandas as pd
import numpy as np
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


def create_target(
    df: pd.DataFrame, target_col: str, group_col: str = "iso3", year_col: str = "year"
) -> pd.DataFrame:
    """Create next-year target by shifting target_col forward within each group.

    Args:
        df: Country-year panel dataframe.
        target_col: Name of the column to shift (e.g., 'NY.GDP.PCAP.KD.ZG').
        group_col: Column to group by (default: 'iso3').
        year_col: Column containing year values (default: 'year').

    Returns:
        Dataframe with added 'target_next_year' column.
    """
    df = df.sort_values([group_col, year_col]).copy()
    df["target_next_year"] = (
        df.groupby(group_col)[target_col].shift(-1)
    )
    n_missing = df["target_next_year"].isna().sum()
    logger.info(
        "Created next-year target from '%s': %d rows, %d missing (last year per country)",
        target_col, len(df), n_missing
    )
    return df


def select_features_by_coverage(
    df: pd.DataFrame,
    feature_cols: List[str],
    min_coverage: float = 0.6,
    train_mask: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Drop features with coverage below min_coverage threshold.

    MINOR FIX: Coverage is computed on training rows only (if train_mask provided)
    to avoid any potential information bleed, even though missingness-based
    selection doesn't leak label signal.

    Args:
        df: Input dataframe.
        feature_cols: List of feature column names to evaluate.
        min_coverage: Minimum fraction of non-null values required (0.0-1.0).
        train_mask: Boolean mask for training rows. If provided, coverage is
                    computed on these rows only.

    Returns:
        Dataframe with low-coverage feature columns removed.
    """
    df = df.copy()
    coverage_df = df[train_mask] if train_mask is not None else df
    dropped = []
    kept = []
    for col in feature_cols:
        if col not in df.columns:
            dropped.append((col, 0.0))
            continue
        coverage = coverage_df[col].notna().mean()
        if coverage >= min_coverage:
            kept.append(col)
        else:
            dropped.append((col, coverage))
            df = df.drop(columns=[col])
    if dropped:
        for col, cov in dropped:
            logger.warning("Dropped feature '%s': coverage %.1f%% < %.1f%%",
                          col, cov * 100, min_coverage * 100)
    logger.info("Features after coverage filter: %d kept, %d dropped",
               len(kept), len(dropped))
    return df


def build_feature_matrix(
    df: pd.DataFrame, feature_cols: List[str], target_col: str = "target_next_year"
) -> Tuple[pd.DataFrame, pd.Series]:
    """Build feature matrix X and target vector y, dropping rows with NaN target.

    Args:
        df: Country-year panel with features and target.
        feature_cols: List of feature column names.
        target_col: Name of the target column.

    Returns:
        Tuple of (X dataframe, y series) with no NaN in target.
    """
    valid_mask = df[target_col].notna()
    df_valid = df[valid_mask].copy()
    X = df_valid[feature_cols].copy()
    y = df_valid[target_col].copy()
    logger.info(
        "Feature matrix: X=%s, y=%s (dropped %d rows with missing target)",
        X.shape, y.shape, (~valid_mask).sum()
    )
    return X, y


def create_temporal_split(
    df: pd.DataFrame,
    train_end: int,
    val_end: int,
    year_col: str = "year",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into chronological train/validation/test sets.

    Args:
        df: Country-year panel.
        train_end: Last year included in training set.
        val_end: Last year included in validation set.
        year_col: Column containing year values.

    Returns:
        Tuple of (train, validation, test) dataframes.
    """
    train = df[df[year_col] <= train_end].copy()
    val = df[(df[year_col] > train_end) & (df[year_col] <= val_end)].copy()
    test = df[df[year_col] > val_end].copy()
    logger.info(
        "Temporal split: train=%d rows (<= %d), val=%d rows (%d-%d), test=%d rows (> %d)",
        len(train), train_end, len(val), train_end + 1, val_end,
        len(test), val_end
    )
    return train, val, test


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    from src.config import load_config
    panel_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/processed/model_data.parquet")
    try:
        panel = pd.read_parquet(panel_path)
        config = load_config()
        panel = create_target(panel, config.target_code)
        feature_cols = [c for c in panel.columns if c not in
                       ["iso3", "country_name", "year", "target_next_year"]]
        # Compute coverage on training rows only
        train_mask = panel["year"] <= config.train_end
        panel = select_features_by_coverage(panel, feature_cols,
                                            min_coverage=0.6, train_mask=train_mask)
        panel.to_parquet(panel_path, index=False)
        logger.info("Updated panel saved to %s", panel_path)
    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        sys.exit(1)
```

- [ ] **Step 4: Run all tests**

Run: `pytest tests/test_features.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/features.py tests/test_features.py
git commit -m "feat: implement feature engineering with train-only coverage"
```

---

## Task 1.3: Write src/visualization.py (MINOR FIX - heatmap averages across features)

**Files:** Create: `src/visualization.py`, Create: `tests/test_visualization.py`

- [ ] **Step 1: Write test_visualization.py**

```python
"""Tests for visualization functions."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from src.visualization import (
    get_project_palette,
    set_project_style,
    plot_missingness_heatmap,
    plot_feature_distributions,
    plot_correlation_matrix,
    plot_actual_vs_predicted,
    plot_residuals,
    plot_feature_importance,
)


def test_get_project_palette_returns_dict():
    """Palette should be a dictionary of named colors."""
    palette = get_project_palette()
    assert isinstance(palette, dict)
    assert "primary" in palette
    assert "secondary" in palette
    assert "accent" in palette


def test_set_project_style_applies_mpl_params():
    """Style function should set matplotlib rcParams."""
    set_project_style()
    assert plt.rcParams["figure.facecolor"] is not None


def test_plot_actual_vs_predicted_returns_figure():
    """Actual vs predicted plot should return a matplotlib figure."""
    fig, ax = plt.subplots()
    result = plot_actual_vs_predicted(
        actual=np.array([1, 2, 3]),
        predicted=np.array([1.1, 2.2, 2.8]),
        ax=ax,
    )
    assert result is not None
    plt.close(fig)


def test_plot_residuals_returns_figure():
    """Residual plot should return a matplotlib figure."""
    fig, ax = plt.subplots()
    result = plot_residuals(
        actual=np.array([1, 2, 3]),
        predicted=np.array([1.1, 2.2, 2.8]),
        ax=ax,
    )
    assert result is not None
    plt.close(fig)


def test_plot_feature_importance_returns_figure():
    """Feature importance plot should return a matplotlib figure."""
    fig, ax = plt.subplots()
    importance = pd.Series({"feat_a": 0.5, "feat_b": 0.3, "feat_c": 0.2})
    result = plot_feature_importance(importance, ax=ax)
    assert result is not None
    plt.close(fig)


def test_plot_missingness_heatmap_averages_across_features():
    """MINOR FIX: Heatmap should show average missingness, not one arbitrary column."""
    df = pd.DataFrame({
        "iso3": ["GHA"] * 3 + ["KEN"] * 3,
        "year": [2018, 2019, 2020] * 2,
        "feat_a": [1.0, np.nan, 3.0, 4.0, 5.0, 6.0],
        "feat_b": [1.0, 2.0, np.nan, np.nan, 5.0, 6.0],
    })
    fig, ax = plt.subplots()
    result = plot_missingness_heatmap(df, ax=ax)
    assert result is not None
    plt.close(fig)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_visualization.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write src/visualization.py (MINOR FIX - heatmap averages across features)**

```python
"""Visualization functions with consistent project visual language.

All charts use the same color palette, font settings, and styling for a
professional, unified look across the application and report.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)

PROJECT_PALETTE = {
    "primary": "#1B4F72",
    "secondary": "#2E86C1",
    "accent": "#E74C3C",
    "positive": "#27AE60",
    "negative": "#C0392B",
    "neutral": "#7F8C8D",
    "background": "#F8F9FA",
    "grid": "#DEE2E6",
    "text": "#2C3E50",
}


def get_project_palette() -> Dict[str, str]:
    """Return the project color palette dictionary.

    Returns:
        Dictionary mapping color names to hex values.
    """
    return PROJECT_PALETTE.copy()


def set_project_style() -> None:
    """Apply consistent matplotlib styling for all project charts."""
    plt.rcParams.update({
        "figure.facecolor": PROJECT_PALETTE["background"],
        "axes.facecolor": "white",
        "axes.edgecolor": PROJECT_PALETTE["grid"],
        "axes.grid": True,
        "grid.color": PROJECT_PALETTE["grid"],
        "grid.alpha": 0.7,
        "text.color": PROJECT_PALETTE["text"],
        "xtick.color": PROJECT_PALETTE["text"],
        "ytick.color": PROJECT_PALETTE["text"],
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "figure.titlesize": 16,
    })
    logger.info("Project matplotlib style applied")


def plot_missingness_heatmap(
    df: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    title: str = "Missing Data by Country and Year",
) -> plt.Figure:
    """Plot a heatmap of average missingness across all feature columns.

    MINOR FIX: Averages missingness across all numeric feature columns
    instead of visualizing one arbitrary column.

    Args:
        df: DataFrame with columns including 'iso3', 'year', and feature columns.
        ax: Matplotlib axes to plot on. If None, creates new figure.
        title: Plot title.

    Returns:
        Matplotlib Figure object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        fig = ax.get_figure()

    # Average missingness across all feature columns per country-year
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                   if c not in ["year"]]
    if not feature_cols:
        logger.warning("No numeric feature columns found for missingness heatmap")
        return fig

    missing_by_group = df.groupby(["iso3", "year"])[feature_cols].apply(
        lambda x: x.isnull().mean().mean()  # Average across features
    ).reset_index(name="avg_missingness")

    pivot = missing_by_group.pivot_table(
        index="iso3", columns="year", values="avg_missingness", aggfunc="first"
    )

    sns.heatmap(
        pivot, ax=ax, cmap="YlOrRd", vmin=0, vmax=1,
        cbar_kws={"label": "Average missing fraction across features"},
        linewidths=0.5,
    )
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel("Country")
    fig.tight_layout()
    return fig


def plot_feature_distributions(
    df: pd.DataFrame,
    feature_cols: list,
    ncols: int = 3,
    title: str = "Feature Distributions",
) -> plt.Figure:
    """Plot histograms of feature distributions.

    Args:
        df: DataFrame containing the features.
        feature_cols: List of feature column names to plot.
        ncols: Number of columns in the subplot grid.
        title: Overall figure title.

    Returns:
        Matplotlib Figure object.
    """
    nrows = (len(feature_cols) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if nrows * ncols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, col in enumerate(feature_cols):
        if col in df.columns:
            axes[i].hist(df[col].dropna(), bins=30, color=PROJECT_PALETTE["secondary"],
                        alpha=0.7, edgecolor="white")
            axes[i].set_title(col, fontsize=10)
            axes[i].set_ylabel("Count")

    for j in range(len(feature_cols), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


def plot_correlation_matrix(
    df: pd.DataFrame,
    feature_cols: list,
    ax: Optional[plt.Axes] = None,
    title: str = "Feature Correlation Matrix",
) -> plt.Figure:
    """Plot a correlation heatmap of features.

    Args:
        df: DataFrame containing the features.
        feature_cols: List of feature column names.
        ax: Matplotlib axes. If None, creates new figure.
        title: Plot title.

    Returns:
        Matplotlib Figure object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.get_figure()

    corr = df[feature_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, ax=ax, cmap="RdBu_r", center=0,
        vmin=-1, vmax=1, annot=True, fmt=".2f",
        square=True, linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_actual_vs_predicted(
    actual: np.ndarray,
    predicted: np.ndarray,
    ax: Optional[plt.Axes] = None,
    title: str = "Actual vs. Predicted GDP Growth",
) -> plt.Figure:
    """Plot actual vs predicted values with identity line.

    Args:
        actual: Array of actual target values.
        predicted: Array of predicted values.
        ax: Matplotlib axes. If None, creates new figure.
        title: Plot title.

    Returns:
        Matplotlib Figure object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.get_figure()

    ax.scatter(actual, predicted, alpha=0.5, color=PROJECT_PALETTE["secondary"],
              edgecolors="white", linewidth=0.5, s=40)

    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())
    ax.plot([min_val, max_val], [min_val, max_val],
           color=PROJECT_PALETTE["accent"], linestyle="--", linewidth=1.5,
           label="Perfect prediction")

    ax.set_xlabel("Actual GDP Growth (%)")
    ax.set_ylabel("Predicted GDP Growth (%)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_residuals(
    actual: np.ndarray,
    predicted: np.ndarray,
    ax: Optional[plt.Axes] = None,
    title: str = "Residual Analysis",
) -> plt.Figure:
    """Plot residuals vs predicted values.

    Args:
        actual: Array of actual target values.
        predicted: Array of predicted values.
        ax: Matplotlib axes. If None, creates new figure.
        title: Plot title.

    Returns:
        Matplotlib Figure object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()

    residuals = actual - predicted
    ax.scatter(predicted, residuals, alpha=0.5, color=PROJECT_PALETTE["secondary"],
              edgecolors="white", linewidth=0.5, s=40)
    ax.axhline(y=0, color=PROJECT_PALETTE["accent"], linestyle="--", linewidth=1.5)
    ax.set_xlabel("Predicted GDP Growth (%)")
    ax.set_ylabel("Residual (Actual - Predicted)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_feature_importance(
    importance: pd.Series,
    ax: Optional[plt.Axes] = None,
    title: str = "Feature Importance",
    top_n: int = 15,
) -> plt.Figure:
    """Plot horizontal bar chart of feature importance.

    Args:
        importance: Series with feature names as index and importance as values.
        ax: Matplotlib axes. If None, creates new figure.
        title: Plot title.
        top_n: Number of top features to display.

    Returns:
        Matplotlib Figure object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    top_features = importance.sort_values(ascending=True).tail(top_n)
    ax.barh(top_features.index, top_features.values,
           color=PROJECT_PALETTE["secondary"], edgecolor="white")
    ax.set_xlabel("Importance")
    ax.set_title(title)
    fig.tight_layout()
    return fig
```

- [ ] **Step 4: Run all tests**

Run: `pytest tests/test_visualization.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/visualization.py tests/test_visualization.py
git commit -m "feat: implement visualization module with project visual language"
```

---

## Task 1.4: Create data profiling notebook

**Files:** Create: `notebooks/01_data_profiling.ipynb`

- [ ] **Step 1: Create notebook**

Structure (AGENTS.md rule 7: Markdown → code → output → interpretation):

1. **Markdown:** "Data Profiling - WDI African Development Indicators"
2. **Markdown:** "## 1. Setup and Data Loading"
3. **Code:** Import pandas, load config, load raw WDI CSV
4. **Code:** `df.shape`, `df.head()`, `df.dtypes`
5. **Markdown:** Brief interpretation of raw data structure
6. **Markdown:** "## 2. African Country Filtering"
7. **Code:** Filter to African countries, log counts
8. **Markdown:** Interpretation: how many countries, which aggregates removed
9. **Markdown:** "## 3. Indicator Coverage Analysis"
10. **Code:** For each indicator: compute % non-null per country and year (on training rows only)
11. **Markdown:** Which indicators pass the 60% threshold, which are dropped
12. **Markdown:** "## 4. Missingness Patterns"
13. **Code:** Heatmap of average missingness by country and year
14. **Markdown:** Interpretation of missingness patterns
15. **Markdown:** "## 5. Feature Distributions"
16. **Code:** Histograms for each indicator
17. **Markdown:** Distribution shapes, skewness, outliers
18. **Markdown:** "## 6. Outlier Inspection"
19. **Code:** Box plots + extreme value tables
20. **Markdown:** Are these real events (crises, hyperinflation) or errors?
21. **Markdown:** "## 7. Correlation Matrix"
22. **Code:** Correlation heatmap
23. **Markdown:** Which features are highly correlated
24. **Markdown:** "## 8. Final Feature Set Decision"
25. **Markdown:** Summary of which features to keep/drop and why

- [ ] **Step 2: Commit**

```bash
git add notebooks/01_data_profiling.ipynb
git commit -m "feat: add data profiling notebook for EDA"
```

---

## Phase 1 Verification

- [ ] `pytest tests/test_data.py tests/test_features.py tests/test_visualization.py -v` ALL PASS
- [ ] Data pipeline can load, filter, reshape WDI data (if raw CSV available)
- [ ] Features selected by coverage computed on training rows only
- [ ] Target created via groupby shift(-1)
- [ ] Temporal split produces correct year boundaries
- [ ] No data leakage: no future information in features
- [ ] No `print()` statements (AGENTS.md rule 4)
- [ ] Errors raised, not swallowed (AGENTS.md rule 5)

---

# Phase 2: Modeling Pipeline

> **Goal:** Train baselines, Ridge, and GradientBoosting models. Evaluate, compare, select, serialize, and write metadata.

---

## Task 2.1: Write src/train.py - Model Training (B2 + B3 FIX)

**Files:** Create: `src/train.py`, Create: `tests/test_train.py`

**B2 FIX:** Persistence baseline now predicts current year's growth (not last training values).
**B3 FIX:** Log transform moved inside pipeline via FunctionTransformer. No separate column creation.

- [ ] **Step 1: Write test_train.py (B2 FIX - persistence test corrected)**

```python
"""Tests for model training, pipelines, and serialization."""
import numpy as np
import pandas as pd
import pytest
import tempfile
from pathlib import Path
from src.train import (
    global_mean_baseline,
    persistence_baseline,
    build_ridge_pipeline,
    build_hgb_pipeline,
    train_and_evaluate,
    save_pipeline,
    load_pipeline,
    compute_metrics,
)


@pytest.fixture
def train_data():
    """Create sample training data."""
    np.random.seed(42)
    X = pd.DataFrame({
        "feat1": np.random.randn(100),
        "feat2": np.random.randn(100),
        "feat3": np.random.randn(100),
    })
    y = pd.Series(np.random.randn(100) * 2 + 5, name="target")
    return X, y


@pytest.fixture
def val_data():
    """Create sample validation data."""
    np.random.seed(99)
    X = pd.DataFrame({
        "feat1": np.random.randn(30),
        "feat2": np.random.randn(30),
        "feat3": np.random.randn(30),
    })
    y = pd.Series(np.random.randn(30) * 2 + 5, name="target")
    return X, y


def test_global_mean_baseline(train_data, val_data):
    """Global mean baseline should predict mean of training target."""
    X_train, y_train = train_data
    X_val, y_val = val_data
    pred = global_mean_baseline(y_train, n_predictions=len(y_val))
    assert len(pred) == len(y_val)
    assert np.allclose(pred, y_train.mean())


def test_persistence_baseline_uses_current_year_growth():
    """B2 FIX: Persistence baseline should predict current year's growth,
    not copy arbitrary historical values."""
    # Simulate: current year growth for test countries
    current_growth = pd.Series([3.2, 5.1, -1.4, 2.8, 4.0])
    result = persistence_baseline(current_growth)
    # Each prediction should equal the corresponding current year value
    np.testing.assert_array_equal(result, current_growth.values)
    assert len(result) == 5


def test_build_ridge_pipeline():
    """Ridge pipeline should have imputer, scaler, and ridge steps."""
    pipeline = build_ridge_pipeline(alpha=1.0)
    assert len(pipeline.named_steps) >= 2
    # Verify it has the right step names
    assert "imputer" in pipeline.named_steps
    assert "model" in pipeline.named_steps


def test_build_hgb_pipeline():
    """HGB pipeline should have imputer and gradient boosting steps."""
    pipeline = build_hgb_pipeline(max_iter=100, random_state=42)
    assert len(pipeline.named_steps) >= 2


def test_ridge_pipeline_handles_nan(train_data):
    """Pipeline should handle NaN values via imputer."""
    X_train, y_train = train_data
    X_train_nan = X_train.copy()
    X_train_nan.iloc[0, 0] = np.nan  # Add a NaN
    pipeline = build_ridge_pipeline()
    pipeline.fit(X_train_nan, y_train)
    pred = pipeline.predict(X_train_nan[:5])
    assert len(pred) == 5
    assert not np.any(np.isnan(pred))


def test_train_and_evaluate_returns_metrics(train_data, val_data):
    """Training should return a dictionary of evaluation metrics."""
    X_train, y_train = train_data
    X_val, y_val = val_data
    pipeline = build_ridge_pipeline()
    metrics = train_and_evaluate(pipeline, X_train, y_train, X_val, y_val)
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics
    assert "directional_accuracy" in metrics
    assert metrics["mae"] >= 0


def test_save_and_load_pipeline(train_data):
    """Pipeline should survive a save/load roundtrip."""
    X_train, y_train = train_data
    pipeline = build_ridge_pipeline()
    pipeline.fit(X_train, y_train)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "model.joblib"
        save_pipeline(pipeline, path)
        loaded = load_pipeline(path)
        pred_original = pipeline.predict(X_train[:5])
        pred_loaded = loaded.predict(X_train[:5])
        np.testing.assert_array_almost_equal(pred_original, pred_loaded)


def test_compute_metrics():
    """Metrics should be computed correctly."""
    actual = np.array([1.0, 2.0, 3.0, 4.0])
    predicted = np.array([1.1, 2.2, 2.8, 3.9])
    metrics = compute_metrics(actual, predicted)
    assert metrics["mae"] == pytest.approx(0.15, abs=0.01)
    assert 0 <= metrics["directional_accuracy"] <= 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_train.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write src/train.py (B2 + B3 FIX)**

```python
"""Model training, evaluation, and serialization.

Defines baseline predictors, sklearn pipelines for Ridge and HistGradientBoosting,
training loops with evaluation, and model artifact saving/loading.

B2 FIX: Persistence baseline predicts current year's growth (not last training values).
B3 FIX: Log transform is applied inside the sklearn pipeline via FunctionTransformer,
        so the app never sees _log columns. The pipeline consumes raw WDI column names.
"""
import numpy as np
import pandas as pd
import joblib
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


def global_mean_baseline(y_train: pd.Series, n_predictions: int) -> np.ndarray:
    """Predict the global training mean for all test observations.

    Args:
        y_train: Training target values.
        n_predictions: Number of predictions to generate.

    Returns:
        Array of constant predictions equal to training mean.
    """
    mean_val = y_train.mean()
    logger.info("Global mean baseline: predicting %.4f for %d observations",
               mean_val, n_predictions)
    return np.full(n_predictions, mean_val)


def persistence_baseline(current_year_growth: pd.Series) -> np.ndarray:
    """Predict next year's growth as this year's observed growth.

    B2 FIX: This is the correct persistence hypothesis:
    "next year's growth = this year's growth." For each test observation,
    the prediction is that row's current-year growth value.

    Args:
        current_year_growth: Current year's observed growth for each test row.

    Returns:
        Array of current year values (used directly as predictions).
    """
    logger.info("Persistence baseline: using current year values for %d predictions",
               len(current_year_growth))
    return current_year_growth.values


def _log1p_transform(X: np.ndarray) -> np.ndarray:
    """Apply log1p transform, clipping negatives to 0 first.

    B3 FIX: This is used inside the pipeline via FunctionTransformer.
    The app feeds raw WDI values; the pipeline handles transformation internally.

    Args:
        X: Input array (may contain negative values).

    Returns:
        log1p-transformed array.
    """
    return np.log1p(np.clip(X, 0, None))


def build_ridge_pipeline(
    alpha: float = 1.0,
    log_transform_features: Optional[List[str]] = None,
    all_feature_names: Optional[List[str]] = None,
) -> Pipeline:
    """Build Ridge regression pipeline with imputation, optional log transform, scaling.

    B3 FIX: Log transform is applied inside the pipeline. If log_transform_features
    is provided, those features get log1p via ColumnTransformer; others pass through.

    Args:
        alpha: Ridge regularization strength.
        log_transform_features: List of feature names to log-transform.
        all_feature_names: Full ordered list of feature names.

    Returns:
        sklearn Pipeline.
    """
    steps = [("imputer", SimpleImputer(strategy="median"))]

    if log_transform_features and all_feature_names:
        # Split features into log-transform and pass-through groups
        log_idx = [all_feature_names.index(f) for f in log_transform_features
                   if f in all_feature_names]
        pass_idx = [i for i in range(len(all_feature_names)) if i not in log_idx]

        preprocessor = ColumnTransformer(
            transformers=[
                ("log", FunctionTransformer(_log1p_transform), log_idx),
                ("pass", "passthrough", pass_idx),
            ],
            remainder="drop",
        )
        steps.append(("log_transform", preprocessor))

    steps.append(("scaler", StandardScaler()))
    steps.append(("model", Ridge(alpha=alpha)))

    pipeline = Pipeline(steps)
    logger.info("Built Ridge pipeline (alpha=%.4f, log_features=%s)",
               alpha, log_transform_features)
    return pipeline


def build_hgb_pipeline(
    max_iter: int = 1000,
    learning_rate: float = 0.05,
    max_depth: int = 5,
    random_state: int = 42,
) -> Pipeline:
    """Build HistGradientBoostingRegressor pipeline with imputation.

    Args:
        max_iter: Maximum boosting iterations.
        learning_rate: Boosting learning rate.
        max_depth: Maximum tree depth.
        random_state: Random seed for reproducibility.

    Returns:
        sklearn Pipeline.
    """
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingRegressor(
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
        )),
    ])
    logger.info(
        "Built HGB pipeline (max_iter=%d, lr=%.4f, depth=%d)",
        max_iter, learning_rate, max_depth
    )
    return pipeline


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute regression evaluation metrics.

    Args:
        y_true: Actual target values.
        y_pred: Predicted values.

    Returns:
        Dictionary with mae, rmse, r2, and directional_accuracy.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    direction_correct = ((y_true >= 0) == (y_pred >= 0)).mean()
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "directional_accuracy": direction_correct,
    }


def train_and_evaluate(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Dict[str, float]:
    """Train a pipeline and evaluate on validation data.

    Args:
        pipeline: sklearn Pipeline to train.
        X_train: Training features.
        y_train: Training target.
        X_val: Validation features.
        y_val: Validation target.

    Returns:
        Dictionary of evaluation metrics on validation set.
    """
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)
    metrics = compute_metrics(y_val.values, y_pred)
    logger.info(
        "Model evaluation - MAE: %.4f, RMSE: %.4f, R2: %.4f, Direction: %.1f%%",
        metrics["mae"], metrics["rmse"], metrics["r2"],
        metrics["directional_accuracy"] * 100
    )
    return metrics


def save_pipeline(pipeline: Pipeline, path: Path) -> None:
    """Save a fitted pipeline to disk using joblib.

    Args:
        pipeline: Fitted sklearn Pipeline.
        path: File path for the saved artifact.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    logger.info("Pipeline saved to %s (%.1f KB)", path, path.stat().st_size / 1024)


def load_pipeline(path: Path) -> Pipeline:
    """Load a fitted pipeline from disk.

    Args:
        path: File path of the saved artifact.

    Returns:
        Fitted sklearn Pipeline.
    """
    pipeline = joblib.load(path)
    logger.info("Pipeline loaded from %s", path)
    return pipeline


def write_model_metadata(
    path: Path,
    feature_names: List[str],
    target_code: str,
    train_end: int,
    val_end: int,
    metrics: Dict[str, float],
    model_type: str,
    random_state: int,
    log_transform_features: Optional[List[str]] = None,
) -> None:
    """Write model metadata JSON with feature contract and metrics.

    B5 FIX: This file is the single source of truth for the model's feature
    contract. The app reads feature names from here, not from config.

    Args:
        path: Path to write model_metadata.json.
        feature_names: Ordered list of feature names the model was trained on.
        target_code: WDI target indicator code.
        train_end: Last training year.
        val_end: Last validation year.
        metrics: Dictionary of evaluation metrics.
        model_type: Name of the model class.
        random_state: Random seed used.
        log_transform_features: Features that get log1p inside the pipeline.
    """
    metadata = {
        "target_code": target_code,
        "target_name": "GDP per capita growth (annual %)",
        "prediction_horizon_years": 1,
        "geographic_scope": "African countries",
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "log_transform_features": log_transform_features or [],
        "train_end": train_end,
        "val_end": val_end,
        "model_type": model_type,
        "random_state": random_state,
        "metrics": metrics,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Model metadata written to %s", path)
```

- [ ] **Step 4: Run all tests**

Run: `pytest tests/test_train.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/train.py tests/test_train.py
git commit -m "feat: implement model training with correct persistence baseline and in-pipeline log transform"
```

---

## Task 2.2: Write src/evaluate.py - Evaluation & Interpretation

**Files:** Create: `src/evaluate.py`, Create: `tests/test_evaluate.py`

- [ ] **Step 1: Write test_evaluate.py**

```python
"""Tests for model evaluation and interpretation."""
import numpy as np
import pandas as pd
import pytest
from src.evaluate import (
    compute_metrics_by_group,
    compute_directional_accuracy,
    compute_bootstrap_ci,
    compute_permutation_importance,
    compute_worst_errors,
)


@pytest.fixture
def sample_predictions():
    """Create sample prediction data for evaluation."""
    np.random.seed(42)
    n = 100
    actual = np.random.randn(n) * 3 + 5
    predicted = actual + np.random.randn(n) * 0.5
    return pd.DataFrame({
        "actual": actual,
        "predicted": predicted,
        "iso3": np.random.choice(["GHA", "KEN", "NGA"], n),
        "year": np.random.choice(range(2018, 2024), n),
    })


def test_compute_metrics_by_group(sample_predictions):
    """Metrics should be computed per group."""
    result = compute_metrics_by_group(sample_predictions, group_col="iso3")
    assert "mae" in result.columns
    assert result["iso3"].nunique() == 3
    assert (result["mae"] >= 0).all()


def test_compute_directional_accuracy(sample_predictions):
    """Directional accuracy should be between 0 and 1."""
    result = compute_directional_accuracy(
        sample_predictions["actual"].values,
        sample_predictions["predicted"].values,
    )
    assert 0 <= result <= 1


def test_compute_bootstrap_ci_returns_bounds(sample_predictions):
    """Bootstrap CI should return lower and upper bounds."""
    lower, upper = compute_bootstrap_ci(
        sample_predictions["actual"].values,
        sample_predictions["predicted"].values,
        metric_fn=lambda a, p: np.mean(np.abs(a - p)),
        n_bootstrap=100,
    )
    assert lower < upper


def test_compute_permutation_importance_returns_series(sample_predictions):
    """Permutation importance should return a Series indexed by feature name."""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer

    X = pd.DataFrame({
        "feat_a": np.random.randn(100),
        "feat_b": np.random.randn(100),
    })
    y = sample_predictions["actual"]
    pipeline = Pipeline([("imputer", SimpleImputer()), ("model", Ridge())])
    pipeline.fit(X, y)

    importance = compute_permutation_importance(
        pipeline, X, y, feature_names=["feat_a", "feat_b"], n_repeats=5
    )
    assert isinstance(importance, pd.Series)
    assert "feat_a" in importance.index


def test_compute_worst_errors(sample_predictions):
    """Worst errors should return top-N rows by absolute error."""
    result = compute_worst_errors(sample_predictions, top_n=5)
    assert len(result) == 5
    assert "abs_error" in result.columns
    assert result["abs_error"].is_monotonic_decreasing
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_evaluate.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write src/evaluate.py**

```python
"""Model evaluation, error analysis, and feature interpretation.

Computes metrics by group, bootstrap confidence intervals, permutation
importance, and worst-error analysis for model comparison.
"""
import numpy as np
import pandas as pd
import logging
from typing import Callable, Tuple, List
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance as sk_permutation_importance
from src.train import compute_metrics

logger = logging.getLogger(__name__)


def compute_metrics_by_group(
    df: pd.DataFrame,
    group_col: str = "iso3",
    actual_col: str = "actual",
    predicted_col: str = "predicted",
) -> pd.DataFrame:
    """Compute evaluation metrics grouped by a column (e.g., country or year).

    Args:
        df: DataFrame with actual and predicted columns.
        group_col: Column to group by.
        actual_col: Column name for actual values.
        predicted_col: Column name for predicted values.

    Returns:
        DataFrame with one row per group and columns for each metric.
    """
    results = []
    for group_name, group_df in df.groupby(group_col):
        metrics = compute_metrics(
            group_df[actual_col].values,
            group_df[predicted_col].values,
        )
        metrics[group_col] = group_name
        results.append(metrics)
    result_df = pd.DataFrame(results)
    logger.info("Computed metrics for %d groups in '%s'", len(result_df), group_col)
    return result_df


def compute_directional_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Compute the fraction of cases where growth direction is correctly predicted.

    Args:
        actual: Array of actual values.
        predicted: Array of predicted values.

    Returns:
        Fraction of correctly predicted directions (0.0 to 1.0).
    """
    correct = ((actual >= 0) == (predicted >= 0)).mean()
    logger.info("Directional accuracy: %.1f%%", correct * 100)
    return correct


def compute_bootstrap_ci(
    actual: np.ndarray,
    predicted: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> Tuple[float, float]:
    """Compute bootstrap confidence interval for a metric.

    Args:
        actual: Array of actual values.
        predicted: Array of predicted values.
        metric_fn: Function that takes (actual, predicted) and returns a scalar.
        n_bootstrap: Number of bootstrap resamples.
        confidence: Confidence level (e.g., 0.95 for 95% CI).
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (lower_bound, upper_bound).
    """
    rng = np.random.RandomState(random_state)
    n = len(actual)
    boot_scores = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        score = metric_fn(actual[idx], predicted[idx])
        boot_scores.append(score)
    boot_scores = np.array(boot_scores)
    alpha = (1 - confidence) / 2
    lower = np.percentile(boot_scores, alpha * 100)
    upper = np.percentile(boot_scores, (1 - alpha) * 100)
    logger.info(
        "Bootstrap CI (%.0f%%): [%.4f, %.4f] (n=%d)",
        confidence * 100, lower, upper, n_bootstrap
    )
    return lower, upper


def compute_permutation_importance(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: List[str],
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.Series:
    """Compute permutation importance for a fitted pipeline.

    Args:
        pipeline: Fitted sklearn Pipeline.
        X: Feature matrix.
        y: Target values.
        feature_names: List of feature names.
        n_repeats: Number of permutation repeats.
        random_state: Random seed.

    Returns:
        Series indexed by feature name with mean importance values.
    """
    result = sk_permutation_importance(
        pipeline, X, y,
        n_repeats=n_repeats,
        random_state=random_state,
    )
    importance = pd.Series(
        result.importances_mean, index=feature_names, name="importance"
    ).sort_values(ascending=False)
    logger.info("Permutation importance computed for %d features", len(importance))
    return importance


def compute_worst_errors(
    df: pd.DataFrame,
    top_n: int = 10,
    actual_col: str = "actual",
    predicted_col: str = "predicted",
) -> pd.DataFrame:
    """Identify the worst prediction errors.

    Args:
        df: DataFrame with actual and predicted columns.
        top_n: Number of worst errors to return.
        actual_col: Column name for actual values.
        predicted_col: Column name for predicted values.

    Returns:
        DataFrame with top_n rows sorted by absolute error descending.
    """
    df = df.copy()
    df["abs_error"] = np.abs(df[actual_col] - df[predicted_col])
    worst = df.nlargest(top_n, "abs_error")
    logger.info("Worst %d errors: MAE=%.4f, max=%.4f", top_n,
               worst["abs_error"].mean(), worst["abs_error"].max())
    return worst
```

- [ ] **Step 4: Run all tests**

Run: `pytest tests/test_evaluate.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/evaluate.py tests/test_evaluate.py
git commit -m "feat: implement evaluation metrics, bootstrap CI, and permutation importance"
```

---

## Task 2.3: Finalize Model & Write Artifacts (B5 FIX - NEW TASK)

**Files:** Modify: `src/train.py` (if needed), Create: `models/growth_model.joblib`, Create: `models/model_metadata.json`

**B5 FIX:** This task was missing. No metadata was ever written, so the deployed app would crash. This task:
1. Selects winner on validation metrics
2. Refits winner on train+val
3. Evaluates once on test
4. Saves: fitted pipeline, model_metadata.json
5. Precomputes permutation importance (avoids recomputing on every Streamlit rerun)

- [ ] **Step 1: Write the training finalization script**

Create `scripts/finalize_model.py`:

```python
"""Finalize model: select winner, refit on train+val, evaluate on test, save artifacts.

This script runs after initial model comparison. It:
1. Loads processed data
2. Trains all candidate models on training data
3. Selects winner by validation MAE
4. Refits winner on train+val combined
5. Evaluates once on test (final reported metrics)
6. Saves: pipeline → growth_model.joblib, metadata → model_metadata.json
7. Precomputes permutation importance for the app
"""
import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.features import create_target, select_features_by_coverage, build_feature_matrix, create_temporal_split
from src.train import (
    build_ridge_pipeline, build_hgb_pipeline, train_and_evaluate,
    save_pipeline, compute_metrics, write_model_metadata, global_mean_baseline,
    persistence_baseline,
)
from src.evaluate import compute_permutation_importance

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    config = load_config()

    # Load processed data
    panel = pd.read_parquet("data/processed/model_data.parquet")
    panel = create_target(panel, config.target_code)

    # Feature selection on training data only
    feature_cols = [c for c in panel.columns if c not in
                   ["iso3", "country_name", "year", "target_next_year"]]
    train_mask = panel["year"] <= config.train_end
    panel = select_features_by_coverage(panel, feature_cols, min_coverage=0.6,
                                        train_mask=train_mask)

    # Determine final feature list
    final_features = [c for c in panel.columns if c not in
                     ["iso3", "country_name", "year", "target_next_year"]]

    # Split
    train, val, test = create_temporal_split(panel, config.train_end, config.val_end)
    X_train, y_train = build_feature_matrix(train, final_features)
    X_val, y_val = build_feature_matrix(val, final_features)
    X_test, y_test = build_feature_matrix(test, final_features)

    logger.info("Train: %s, Val: %s, Test: %s", X_train.shape, X_val.shape, X_test.shape)

    # Determine log transform features from config
    log_features = [f for f in config.log_transform_candidates if f in final_features]

    # Train candidates
    ridge = build_ridge_pipeline(alpha=config.ridge_alpha,
                                 log_transform_features=log_features,
                                 all_feature_names=final_features)
    hgb = build_hgb_pipeline(max_iter=config.hgb_max_iter,
                             learning_rate=config.hgb_learning_rate,
                             max_depth=config.hgb_max_depth,
                             random_state=config.random_state)

    ridge_metrics = train_and_evaluate(ridge, X_train, y_train, X_val, y_val)
    hgb_metrics = train_and_evaluate(hgb, X_train, y_train, X_val, y_val)

    logger.info("Ridge val MAE: %.4f", ridge_metrics["mae"])
    logger.info("HGB val MAE: %.4f", hgb_metrics["mae"])

    # Select winner
    if hgb_metrics["mae"] <= ridge_metrics["mae"]:
        winner_name = "HistGradientBoostingRegressor"
        winner_pipeline = hgb
        winner_val_metrics = hgb_metrics
    else:
        winner_name = "Ridge"
        winner_pipeline = ridge
        winner_val_metrics = ridge_metrics

    logger.info("Winner: %s (val MAE: %.4f)", winner_name, winner_val_metrics["mae"])

    # Refit winner on train+val
    X_trainval = pd.concat([X_train, X_val])
    y_trainval = pd.concat([y_train, y_val])
    winner_pipeline.fit(X_trainval, y_trainval)

    # Evaluate on test (final metrics)
    y_pred_test = winner_pipeline.predict(X_test)
    test_metrics = compute_metrics(y_test.values, y_pred_test)
    logger.info("Test metrics: %s", test_metrics)

    # Baselines for comparison
    gm_pred = global_mean_baseline(y_train, len(y_test))
    gm_metrics = compute_metrics(y_test.values, gm_pred)
    logger.info("Global mean baseline test MAE: %.4f", gm_metrics["mae"])

    # Persistence: for test set, prediction = current year's growth
    persistence_growth = test["NY.GDP.PCAP.KD.ZG"].values
    valid_persistence = ~np.isnan(persistence_growth)
    if valid_persistence.any():
        pers_pred = persistence_baseline(pd.Series(persistence_growth[valid_persistence]))
        pers_metrics = compute_metrics(
            y_test.values[valid_persistence], pers_pred
        )
        logger.info("Persistence baseline test MAE: %.4f", pers_metrics["mae"])

    # Save pipeline
    save_pipeline(winner_pipeline, Path("models/growth_model.joblib"))

    # Precompute permutation importance for the app
    importance = compute_permutation_importance(
        winner_pipeline, X_test, y_test, final_features, n_repeats=10
    )

    # Write metadata
    all_metrics = {
        "global_mean_baseline": gm_metrics,
        "persistence_baseline": pers_metrics if valid_persistence.any() else {},
        "ridge_val": ridge_metrics,
        "hgb_val": hgb_metrics,
        "winner_test": test_metrics,
        "feature_importance": importance.to_dict(),
    }

    write_model_metadata(
        path=Path("models/model_metadata.json"),
        feature_names=final_features,
        target_code=config.target_code,
        train_end=config.train_end,
        val_end=config.val_end,
        metrics=all_metrics,
        model_type=winner_name,
        random_state=config.random_state,
        log_transform_features=log_features,
    )

    # Save country metadata
    country_meta = panel[["iso3", "country_name"]].drop_duplicates().sort_values("iso3")
    country_meta.to_csv("data/processed/country_metadata.csv", index=False)
    logger.info("Country metadata saved (%d countries)", len(country_meta))

    logger.info("All artifacts saved. Ready for deployment.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the finalization script**

Run: `python scripts/finalize_model.py`
Expected: Logs show winner selection, test metrics, artifacts saved

- [ ] **Step 3: Verify artifacts exist**

```bash
ls -la models/growth_model.joblib models/model_metadata.json
```

- [ ] **Step 4: Commit**

```bash
git add scripts/finalize_model.py models/growth_model.joblib models/model_metadata.json data/processed/country_metadata.csv
git commit -m "feat: finalize model - select winner, save pipeline and metadata"
```

---

## Task 2.4: Create model evaluation notebook (with bootstrap CI - GAP FIX)

**Files:** Create: `notebooks/02_model_evaluation.ipynb`

**GAP FIX:** Bootstrap CI is now invoked in the notebook.

- [ ] **Step 1: Create notebook**

Structure:

1. **Markdown:** "Model Evaluation - Africa Growth Explorer"
2. **Code:** Load processed data, trained pipeline, metadata
3. **Markdown:** "## 1. Baseline Comparison"
4. **Code:** Run global mean + persistence baselines, compute metrics
5. **Markdown:** "## 2. Ridge Regression Results"
6. **Code:** Train Ridge, evaluate on val
7. **Markdown:** "## 3. Gradient Boosting Results"
8. **Code:** Train HGB, evaluate on val
9. **Markdown:** "## 4. Model Comparison Table"
10. **Code:** Side-by-side metrics table (val + test)
11. **Markdown:** "## 5. Actual vs. Predicted"
12. **Code:** Scatter plot for best model
13. **Markdown:** "## 6. Residual Analysis"
14. **Code:** Residual plots
15. **Markdown:** "## 7. Feature Importance"
16. **Code:** Ridge coefficients + HGB permutation importance (from metadata)
17. **Markdown:** "## 8. Bootstrap Confidence Intervals"
18. **Code:** Bootstrap 95% CI for MAE, RMSE, directional accuracy on test
19. **Markdown:** Interpretation of CIs
20. **Markdown:** "## 9. Error Analysis"
21. **Code:** Worst errors, country-level metrics
22. **Markdown:** "## 10. COVID-19 Placement Note"
23. **Markdown:** Explanation that 2020 shock sits in validation, test is post-COVID 2021+. This is a deliberate decision.

- [ ] **Step 2: Commit**

```bash
git add notebooks/02_model_evaluation.ipynb
git commit -m "feat: add model evaluation notebook with bootstrap CI"
```

---

## Phase 2 Verification

- [ ] `pytest tests/test_train.py tests/test_evaluate.py -v` ALL PASS
- [ ] `python scripts/finalize_model.py` runs successfully
- [ ] `models/growth_model.joblib` exists
- [ ] `models/model_metadata.json` exists with feature_names, metrics, split years
- [ ] `data/processed/country_metadata.csv` exists
- [ ] Persistence baseline correctly predicts current year's growth
- [ ] Log transform is inside pipeline (no `_log` columns in training data)
- [ ] Feature list in metadata matches what pipeline was trained on
- [ ] No data leakage (AGENTS.md rule 6)
- [ ] No `print()` statements (AGENTS.md rule 4)

---

# Phase 3: Application & Deployment

> **Goal:** Build Streamlit app that reads from metadata, deploy to Streamlit Cloud.

---

## Task 3.1: Write app.py - Streamlit Application (B3 + B4 + B5 FIX)

**Files:** Create: `app.py`

**B3 FIX:** App reads feature list from `model_metadata.json`, not from config. Always passes full feature list (NaNs handled by pipeline imputer).
**B4 FIX:** Slider range is data min/max (full range), warning fires on P1-P99 band.
**B5 FIX:** App loads `model_metadata.json` which now exists (Task 2.3 creates it). Permutation importance loaded from metadata, not recomputed.

- [ ] **Step 1: Write app.py**

```python
"""Africa Growth Explorer - Streamlit Decision-Support Application.

Loads a pre-trained sklearn pipeline and processed data to provide:
- Project overview with methodology description
- Country-level exploration of development indicators
- Model performance visualization
- Interactive scenario explorer with extrapolation warnings

B3 FIX: Feature list is read from model_metadata.json (single source of truth).
B4 FIX: Slider range is data min/max; warning fires on P1-P99 band.
B5 FIX: Permutation importance loaded from metadata, not recomputed on rerun.
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import logging
from pathlib import Path
from src.config import load_config, Config
from src.visualization import (
    set_project_style,
    get_project_palette,
    plot_actual_vs_predicted,
    plot_residuals,
    plot_feature_importance,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Africa Growth Explorer",
    page_icon="🌍",
    layout="wide",
)
set_project_style()


@st.cache_resource
def load_model(path: str = "models/growth_model.joblib"):
    """Load the trained sklearn pipeline (cached across reruns)."""
    return joblib.load(Path(path))


@st.cache_data
def load_processed_data(path: str = "data/processed/model_data.parquet"):
    """Load the processed country-year panel (cached across reruns)."""
    return pd.read_parquet(Path(path))


@st.cache_data
def load_model_metadata(path: str = "models/model_metadata.json"):
    """Load model metadata - single source of truth for feature contract.

    B5 FIX: This file defines which features the model expects,
    in what order, and what metrics it achieved.
    """
    with open(Path(path), encoding="utf-8") as f:
        return json.load(f)


def page_overview():
    """Render the project overview page."""
    st.title("🌍 Africa Growth Explorer")
    st.markdown("""
    ### Machine Learning Decision-Support System

    **Core Question:** To what extent can recent development indicators predict
    near-term GDP per capita growth across African countries?

    **Data Source:** World Bank World Development Indicators (WDI)

    **Models:** Ridge Regression + HistGradientBoostingRegressor

    **Validation:** Chronological train/val/test split (no data leakage)

    **Important:** This system provides *predictive associations*, not causal estimates.
    Scenario results show how the model responds to different indicator values and
    should not be interpreted as policy recommendations.
    """)

    st.subheader("How It Works")
    st.markdown("""
    1. **Data**: Development indicators for African countries (2000-present)
    2. **Model**: Predicts next-year GDP per capita growth from current indicators
    3. **Validation**: Train (2000-2017), Validation (2018-2020), Test (2021+)
    4. **Application**: Explore countries, view predictions, test scenarios
    """)

    st.warning(
        "⚠️ **Causal Interpretation Disclaimer**: Scenario results show how the "
        "predictive model responds to alternative indicator values. They should "
        "NOT be interpreted as causal estimates of the effect of implementing "
        "a specific policy."
    )


def page_explore(data: pd.DataFrame):
    """Render the country exploration page.

    MINOR FIX: Guard against IndexError when a country has no non-null growth values.
    """
    st.title("📊 Explore Africa")

    countries = sorted(data["country_name"].unique())
    selected_country = st.selectbox("Select a country", countries)

    country_data = data[data["country_name"] == selected_country].sort_values("year")

    st.subheader(f"{selected_country} - Growth Trend")
    growth_data = country_data.dropna(subset=["NY.GDP.PCAP.KD.ZG"])
    if growth_data.empty:
        st.info("No GDP growth data available for this country.")
    else:
        st.line_chart(growth_data.set_index("year")["NY.GDP.PCAP.KD.ZG"])

    st.subheader("Key Indicators Over Time")
    indicator_cols = [c for c in country_data.columns
                     if c not in ["iso3", "country_name", "year", "target_next_year"]]
    selected_indicators = st.multiselect(
        "Select indicators to display",
        indicator_cols,
        default=indicator_cols[:3] if len(indicator_cols) >= 3 else indicator_cols,
    )
    if selected_indicators:
        st.line_chart(country_data.set_index("year")[selected_indicators])

    st.subheader("Latest Available Data")
    latest = country_data.dropna(subset=["NY.GDP.PCAP.KD.ZG"])
    if not latest.empty:
        st.dataframe(latest.iloc[-1:].to_frame().T)
    else:
        st.info("No data available.")


def page_performance(data: pd.DataFrame, model, metadata: dict):
    """Render the model performance page.

    B5 FIX: Permutation importance loaded from metadata, not recomputed.
    """
    st.title("📈 Model Performance")

    feature_names = metadata["feature_names"]
    metrics = metadata["metrics"]

    # Display test metrics from metadata
    winner_metrics = metrics.get("winner_test", {})
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MAE", f"{winner_metrics.get('mae', 0):.3f}")
    col2.metric("RMSE", f"{winner_metrics.get('rmse', 0):.3f}")
    col3.metric("R²", f"{winner_metrics.get('r2', 0):.3f}")
    col4.metric("Directional Accuracy",
                f"{winner_metrics.get('directional_accuracy', 0):.1%}")

    # Baseline comparison
    st.subheader("Baseline Comparison")
    gm = metrics.get("global_mean_baseline", {})
    pers = metrics.get("persistence_baseline", {})
    baseline_df = pd.DataFrame({
        "Model": ["Global Mean", "Persistence", "Winner (Test)"],
        "MAE": [gm.get("mae", 0), pers.get("mae", 0), winner_metrics.get("mae", 0)],
        "Directional Acc": [
            gm.get("directional_accuracy", 0),
            pers.get("directional_accuracy", 0),
            winner_metrics.get("directional_accuracy", 0),
        ],
    })
    st.dataframe(baseline_df)

    # Actual vs Predicted
    st.subheader("Actual vs. Predicted (Test Set)")
    from src.features import create_target, build_feature_matrix, create_temporal_split
    config = load_config()
    panel = data.copy()
    panel = create_target(panel, config.target_code)
    _, _, test = create_temporal_split(panel, config.train_end, config.val_end)
    X_test = test[feature_names]
    y_test = test["target_next_year"]
    valid = y_test.notna()
    X_test, y_test = X_test[valid], y_test[valid]
    y_pred = model.predict(X_test)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    plot_actual_vs_predicted(y_test.values, y_pred, ax=ax)
    st.pyplot(fig)
    plt.close(fig)

    # Residuals
    st.subheader("Residual Analysis")
    fig, ax = plt.subplots()
    plot_residuals(y_test.values, y_pred, ax=ax)
    st.pyplot(fig)
    plt.close(fig)

    # Feature importance (B5 FIX: from metadata, not recomputed)
    st.subheader("Feature Importance (Permutation)")
    importance_dict = metrics.get("feature_importance", {})
    if importance_dict:
        importance = pd.Series(importance_dict).sort_values(ascending=False)
        fig, ax = plt.subplots()
        plot_feature_importance(importance, ax=ax)
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Feature importance not available in metadata.")


def page_scenario(data: pd.DataFrame, model, metadata: dict, config: Config):
    """Render the interactive scenario explorer page.

    B3 FIX: Always passes full feature list from metadata. NaNs handled by pipeline.
    B4 FIX: Slider range is data min/max (full range). Warning fires on P1-P99 band.
    """
    st.title("🔮 Scenario Explorer")

    st.warning(
        "⚠️ **Causal Interpretation Disclaimer**: Scenario results show how the "
        "predictive model responds to alternative indicator values. They should "
        "NOT be interpreted as causal estimates of the effect of implementing "
        "a specific policy."
    )

    feature_names = metadata["feature_names"]

    countries = sorted(data["country_name"].unique())
    selected_country = st.selectbox("Select a country", countries, key="scenario_country")

    country_data = data[data["country_name"] == selected_country].sort_values("year")
    years = sorted(country_data["year"].unique())
    selected_year = st.selectbox("Select a reference year", years, index=len(years)-1)

    current = country_data[country_data["year"] == selected_year].iloc[0]

    st.subheader("Current Indicator Values")
    display_features = [f for f in feature_names if f in current.index]
    current_display = current[display_features].to_frame().T
    st.dataframe(current_display)

    # Scenario controls (B4 FIX: slider range is data min/max, not P1-P99)
    st.subheader("Adjust Scenario Indicators")
    # Pick scenario variables from the feature list (3-5 with understandable units)
    scenario_candidates = ["EG.ELC.ACCS.ZS", "IT.NET.USER.ZS", "NE.GDI.TOTL.ZS",
                          "NE.TRD.GNFS.ZS", "FP.CPI.TOTL.ZG"]
    scenario_vars = [v for v in scenario_candidates if v in feature_names]

    # Compute training-data statistics for bounds and warnings
    train_data = data[data["year"] <= config.train_end]

    scenario_values = {}
    for var in scenario_vars:
        feature_name = next((f.name for f in config.features if f.code == var), var)
        current_val = float(current[var]) if pd.notna(current[var]) else 0.0

        # B4 FIX: Slider range is data min/max (full observed range)
        if var in train_data.columns:
            obs_min = float(train_data[var].min())
            obs_max = float(train_data[var].max())
            p01 = float(train_data[var].quantile(0.01))
            p99 = float(train_data[var].quantile(0.99))
        else:
            obs_min, obs_max = current_val * 0.5, current_val * 2.0
            p01, p99 = obs_min, obs_max

        # Ensure current value is within slider range
        slider_min = min(obs_min, current_val)
        slider_max = max(obs_max, current_val)

        scenario_values[var] = {
            "value": st.slider(
                feature_name,
                min_value=slider_min,
                max_value=slider_max,
                value=current_val,
                step=(slider_max - slider_min) / 100 if slider_max > slider_min else 0.1,
            ),
            "p01": p01,
            "p99": p99,
        }

    # Generate predictions
    if st.button("Generate Scenario Prediction"):
        # B3 FIX: Always pass full feature list from metadata
        # NaNs are fine - the pipeline's SimpleImputer handles them
        baseline_input = pd.DataFrame([{f: current.get(f, np.nan) for f in feature_names}])
        baseline_pred = model.predict(baseline_input)[0]

        scenario_input = baseline_input.copy()
        for var, info in scenario_values.items():
            scenario_input[var] = info["value"]
        scenario_pred = model.predict(scenario_input)[0]

        col1, col2, col3 = st.columns(3)
        col1.metric("Baseline Prediction", f"{baseline_pred:.2f}%")
        col2.metric("Scenario Prediction", f"{scenario_pred:.2f}%")
        diff = scenario_pred - baseline_pred
        col3.metric("Difference", f"{diff:+.2f} pp",
                    delta=f"{'+' if diff > 0 else ''}{diff:.2f} pp")

        # B4 FIX: Extrapolation warning fires on P1-P99 band
        for var, info in scenario_values.items():
            val = info["value"]
            p01 = info["p01"]
            p99 = info["p99"]
            feature_name = next((f.name for f in config.features if f.code == var), var)
            if val < p01 or val > p99:
                st.warning(
                    f"⚠️ **{feature_name}** value ({val:.2f}) is outside the "
                    f"P1-P99 range [{p01:.2f}, {p99:.2f}]. "
                    f"The result may be unreliable (extrapolation)."
                )


def main():
    """Main application entry point."""
    try:
        model = load_model()
        data = load_processed_data()
        metadata = load_model_metadata()
    except FileNotFoundError as e:
        st.error(f"Required artifact not found: {e}")
        st.info("Run `python scripts/finalize_model.py` first to generate artifacts.")
        st.stop()

    config = load_config()

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Project Overview", "Explore Africa", "Model Performance", "Scenario Explorer"],
    )

    if page == "Project Overview":
        page_overview()
    elif page == "Explore Africa":
        page_explore(data)
    elif page == "Model Performance":
        page_performance(data, model, metadata)
    elif page == "Scenario Explorer":
        page_scenario(data, model, metadata, config)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Manual verification**

Run: `streamlit run app.py`
Expected: App loads in browser with 4 pages, no errors

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: implement Streamlit app - reads from metadata, correct persistence, working guardrails"
```

---

## Task 3.2: Write README.md

**Files:** Modify: `README.md`

- [ ] **Step 1: Write README.md**

```markdown
# Africa Growth Explorer

A machine learning decision-support system for predicting near-term GDP per capita growth across African countries using World Bank Development Indicators.

## Problem Statement

**Core Question:** To what extent can recent development indicators predict near-term GDP per capita growth across African countries?

**Decision-Support Question:** Given a country's current development profile, what level of next-year GDP per capita growth does the model estimate, and which indicators contribute most?

## Data Source

- **Source:** [World Bank World Development Indicators](https://datatopics.worldbank.org/world-development-indicators/)
- **Scope:** African countries only
- **Time Range:** 2000 - latest available year
- **Features:** 14 development indicators (infrastructure, health, education, etc.)
- **Target:** GDP per capita growth (annual %) at t+1

## Project Architecture

```
africa-growth-ml/
├── app.py                    # Streamlit application
├── config/indicators.yaml    # Feature definitions and settings
├── data/processed/           # Processed datasets (committed)
├── models/                   # Serialized pipelines + metadata (committed)
│   ├── growth_model.joblib
│   └── model_metadata.json   # Feature contract + metrics
├── notebooks/                # EDA and evaluation notebooks
├── src/                      # Reusable source modules
│   ├── config.py
│   ├── data.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   └── visualization.py
├── scripts/
│   └── finalize_model.py     # Model selection and artifact creation
└── tests/                    # Unit tests
```

## Setup

```bash
pip install -r requirements.txt
```

## How to Run

1. Download WDI data (see `data/README.md`)
2. Run the data pipeline:
   ```bash
   python -m src.data
   python -m src.features
   ```
3. Finalize the model:
   ```bash
   python scripts/finalize_model.py
   ```
4. Launch the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Model Evaluation

| Model | MAE | RMSE | Directional Accuracy |
|-------|-----|------|---------------------|
| Global Mean Baseline | - | - | - |
| Persistence Baseline | - | - | - |
| Ridge Regression | - | - | - |
| Gradient Boosting | - | - | - |

*Values populated after training.*

## Limitations

- **Not causal:** Model identifies associations, not causal effects
- **Temporal scope:** Predicts short-term (1-year) growth only
- **Geographic scope:** African countries only
- **Data dependency:** Performance depends on WDI data quality and availability
- **COVID-19:** 2020 shock is in validation set; test is post-COVID 2021+

## Causal Interpretation Disclaimer

This application provides predictive analysis, not causal policy recommendations. Scenario results show how the model responds to different indicator values and should not be interpreted as estimates of what would happen if a specific policy were implemented.

## License

MIT License
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add comprehensive README"
```

---

## Task 3.3: Deploy to Streamlit Cloud

- [ ] **Step 1: Create .streamlit/config.toml**

```toml
[theme]
primaryColor = "#1B4F72"
backgroundColor = "#F8F9FA"
secondaryBackgroundColor = "#FFFFFF"
textColor = "#2C3E50"
```

- [ ] **Step 2: Verify deployment checklist**

- [ ] `app.py` is at repository root
- [ ] All imports work from root
- [ ] Model files are committed (NOT gitignored)
- [ ] `requirements.txt` installs successfully
- [ ] No local absolute paths
- [ ] No live API calls
- [ ] Python 3.11 selected in Streamlit Cloud advanced settings

- [ ] **Step 3: Push to GitHub and connect to Streamlit Cloud**

- [ ] **Step 4: Commit**

```bash
git add .streamlit/
git commit -m "chore: add Streamlit Cloud configuration"
```

---

## Phase 3 Verification

- [ ] `streamlit run app.py` loads without errors
- [ ] All 4 pages render correctly
- [ ] Country selection works
- [ ] Model predictions generate correctly
- [ ] Scenario explorer: sliders use full data range, extrapolation warnings fire
- [ ] Causal disclaimer visible on relevant pages
- [ ] Feature list from metadata matches pipeline
- [ ] Deployed app loads model and makes predictions

---

# Phase 4: Deliverables

> **Goal:** Write the capstone report and presentation slides outline.

---

## Task 4.1: Write Capstone Report (GAP FIX)

**Files:** Create: `reports/capstone_report.md` (to be converted to PDF)

**GAP FIX:** This task was missing. The report is a graded deliverable.

- [ ] **Step 1: Write report in markdown**

```markdown
# Africa Growth Explorer: A Machine Learning Decision-Support System

## 1. Introduction

African economic development remains a critical area for research and policy analysis.
Understanding which development conditions predict near-term growth can help analysts
and researchers prioritize their investigations.

Machine learning offers a way to identify complex, non-linear relationships between
multiple development indicators and future economic performance. This project builds
a predictive system using World Bank World Development Indicators (WDI) to estimate
next-year GDP per capita growth for African countries.

The goal is not to prove causation but to provide a screening tool that identifies
which development profiles are associated with higher or lower predicted growth.

## 2. Problem Statement

**Prediction task:** Regression - predict GDP per capita growth (annual %) for year t+1
using development indicators observed during year t.

**Target:** `NY.GDP.PCAP.KD.ZG` (GDP per capita growth, annual %)

**Time horizon:** One year ahead

**Geographic scope:** African countries only (approximately 54 countries)

**Intended users:** Development analysts, economic researchers, policy analysts,
government planning teams, NGOs

**Expected impact:** A screening tool that helps analysts identify which countries
and indicators to investigate further, not a tool for making final policy decisions.

## 3. Dataset Description

- **Source:** World Bank World Development Indicators (WDI)
- **URL:** https://datatopics.worldbank.org/world-development-indicators/
- **Indicators considered:** 14 candidate features + 1 target
- **Time range:** 2000 - latest available year
- **Geographic filter:** African countries (via WDI metadata region classification)
- **Final feature set:** 8-12 indicators (determined by coverage analysis)
- **Target definition:** GDP per capita growth at t+1 (groupby country, shift(-1))
- **Data limitations:** Missing data varies by country and indicator; some indicators
  have poor coverage for earlier years

## 4. Methodology

### Data Transformation
- Raw WDI CSV (wide format) reshaped to country-year panel (long format)
- Indicators pivoted into feature columns
- Numeric conversion with placeholder handling

### Missing-Data Strategy
- Coverage analysis computed on training rows only
- Features with <60% coverage dropped
- Rows with missing target dropped
- Remaining missing values handled by median imputation inside sklearn pipeline

### Outlier Handling
- Outliers investigated but not automatically removed
- Extreme values (hyperinflation, crises) retained as genuine observations
- No robust clipping applied

### Feature Engineering
- Log transform applied inside pipeline (FunctionTransformer) for GDP per capita levels
- No separate _log columns created

### Temporal Split
- Training: 2000-2017
- Validation: 2018-2020
- Test: 2021+

Note: The 2020 COVID-19 shock falls in the validation period. This means the model
selection process is informed by how models handle a major shock year. The test set
is entirely post-COVID, which tests generalization after the shock.

### Baselines
- **Global mean:** Predict training-period mean for all test observations
- **Persistence:** Predict next year's growth as this year's growth

### Models
- **Ridge Regression:** Pipeline: SimpleImputer(median) → StandardScaler → Ridge
- **HistGradientBoostingRegressor:** Pipeline: SimpleImputer(median) → HGBR

### Evaluation Metrics
- **MAE** (primary): Mean Absolute Error in percentage points
- **RMSE:** Root Mean Squared Error
- **R²:** Coefficient of determination
- **Directional accuracy:** Fraction of cases where growth direction is correct

## 5. EDA Findings

- [Summarize findings from 01_data_profiling.ipynb]
- Missingness patterns by country and year
- Feature distributions and correlations
- Outlier inspection results
- Final feature selection rationale

## 6. Model Development

### Ridge Regression
- Regularized linear model robust to correlated indicators
- Pipeline ensures imputer is fitted on training data only
- Alpha selected via grid search on expanding-window validation

### HistGradientBoostingRegressor
- Captures non-linear relationships and feature interactions
- Hist-based implementation is memory-efficient
- Max iterations, learning rate, and depth tuned on validation

### Hyperparameter Tuning
- Compact grid search (not large randomized search)
- Expanding-window validation within training period
- Selected by validation MAE + stability across folds

## 7. Evaluation Results

| Model | MAE | RMSE | R² | Directional Accuracy |
|-------|-----|------|-----|---------------------|
| Global Mean Baseline | [value] | [value] | [value] | [value] |
| Persistence Baseline | [value] | [value] | [value] | [value] |
| Ridge Regression | [value] | [value] | [value] | [value] |
| Gradient Boosting | [value] | [value] | [value] | [value] |

### Additional Analysis
- Actual vs. predicted scatter plot
- Residual analysis
- Bootstrap 95% confidence intervals for MAE, RMSE, directional accuracy
- Performance by country
- Performance by year
- Worst prediction errors
- Feature importance comparison (Ridge coefficients vs. HGB permutation importance)

## 8. Interpretation

### Global Feature Importance
- [List top features from permutation importance]
- [Compare Ridge coefficient directions with HGB importance]

### Country-Specific Patterns
- [Highlight any countries with consistently high/low errors]

### Why Feature Importance Does Not Causality
- Feature importance shows association, not causation
- Confounding variables may drive both the feature and the target
- Changing one indicator does not necessarily change the prediction
- Causal claims would require experimental or quasi-experimental methods

## 9. Decision-Support Application

### Streamlit Architecture
- Single-page app with 4 tabs: Overview, Explore, Performance, Scenario
- Pre-trained pipeline loaded via joblib
- No backend API - inference runs directly on Streamlit server

### Scenario Explorer
- User selects country and reference year
- 3-5 adjustable indicators with slider controls
- Baseline vs. scenario prediction comparison
- Extrapolation warnings when values outside P1-P99 band

### Intended Use
- Initial country screening
- Identifying indicators that warrant deeper investigation
- Comparing development profiles across countries
- NOT for making final policy decisions

## 10. Causal Limitations

- **Confounding:** Unobserved variables may influence both indicators and growth
- **Reverse causality:** Growth itself affects development indicators
- **Omitted variables:** Important factors not included in the model
- **Measurement error:** WDI data quality varies by country and reporting year
- **Country heterogeneity:** Model may not capture unique country-specific dynamics

Predictive scenarios are not causal interventions. Changing an indicator value
in the app shows what the model associates with different profiles, not what
would happen if a policy changed that indicator.

## 11. Recommendations

1. Use the dashboard for initial country screening, not final decisions
2. Investigate consistently important indicators through domain research
3. Combine model output with expert knowledge and country-specific evidence
4. Do not allocate funding solely from model predictions
5. Use causal research before treating any indicator as a policy lever
6. Re-train periodically as WDI data and economic conditions change
7. Monitor performance during structural shocks

## 12. Conclusion

This project built an end-to-end ML decision-support system for predicting
near-term GDP per capita growth across African countries. The system:

- Processes World Bank WDI data into a clean country-year panel
- Trains and compares Ridge Regression and Gradient Boosting models
- Validates using chronological splits to prevent data leakage
- Deploys as a Streamlit Cloud application with scenario exploration
- Provides responsible causal interpretation guidance

The Gradient Boosting model [outperformed/did not outperform] Ridge Regression
on validation metrics, with [model] achieving the best test-set performance.

Main limitations: the model identifies associations, not causal effects;
performance varies by country and time period; and the system should be
used alongside expert judgment, not as a replacement for it.
```

- [ ] **Step 2: Commit**

```bash
git add reports/capstone_report.md
git commit -m "docs: add capstone report draft (to be converted to PDF)"
```

---

## Task 4.2: Write Presentation Slides Outline (GAP FIX)

**Files:** Create: `presentation/slides_outline.md`

**GAP FIX:** Presentation content written as markdown so user only needs to add visuals.

- [ ] **Step 1: Write slides outline**

```markdown
# Africa Growth Explorer - Presentation Outline

**Duration:** 5-10 minutes
**Format:** Slide deck (content below, user builds visuals)

---

## Slide 1: Title

**Title:** Africa Growth Explorer
**Subtitle:** A Machine Learning Decision-Support System for Predicting GDP Per Capita Growth
**Your name, date, program**

---

## Slide 2: Problem Statement

**Heading:** The Problem

**Content:**
- African development analysis needs data-driven screening tools
- Current approaches rely on individual indicators or descriptive statistics
- Question: Can multiple development indicators predict near-term GDP growth?

**Key point:** We built a predictive system, not a causal model.

---

## Slide 3: Dataset

**Heading:** Data Source

**Content:**
- World Bank World Development Indicators (WDI)
- 14 development indicators across African countries
- Time range: 2000 - present
- Target: GDP per capita growth at t+1

**Visual:** Map of Africa with country coverage, or bar chart of indicator availability

---

## Slide 4: Analysis Process

**Heading:** Methodology

**Content:**
- Data pipeline: Raw WDI → country-year panel → feature selection
- Coverage-based feature selection (60% threshold)
- Chronological train/val/test split (no data leakage)
- Two models: Ridge Regression + Gradient Boosting

**Visual:** Pipeline flow diagram

---

## Slide 5: Model Performance

**Heading:** Results

**Content:**
- Comparison table: Baselines vs. Ridge vs. Gradient Boosting
- Primary metric: MAE (percentage points)
- Directional accuracy: Can the model predict growth direction?

**Visual:** Actual vs. predicted scatter plot, metrics table

---

## Slide 6: Key Insights

**Heading:** What We Learned

**Content:**
- Top predictive indicators (from permutation importance)
- Which development conditions are most informative
- How linear vs. non-linear models differ in their interpretations
- Error patterns: which countries/years are hardest to predict

**Visual:** Feature importance bar chart

---

## Slide 7: Scenario Explorer Demo

**Heading:** Decision-Support Application

**Content:**
- Streamlit Cloud deployment
- Country selection and indicator exploration
- Scenario analysis with extrapolation warnings
- Causal disclaimer prominently displayed

**Visual:** Screenshot of the scenario explorer

---

## Slide 8: Recommendations

**Heading:** Recommendations

**Content:**
- Use for screening, not final decisions
- Combine with domain expertise and country-specific evidence
- Investigate consistently important indicators through research
- Do not allocate resources based solely on model predictions
- Re-train as new data becomes available

---

## Slide 9: Limitations & Future Work

**Heading:** Limitations

**Content:**
- Association, not causation
- Performance varies by country and time period
- Limited to available WDI indicators
- Future: prediction intervals, more granular regional analysis, additional data sources

---

## Slide 10: Thank You

**Heading:** Thank You

**Content:**
- Questions?
- Dashboard link
- Repository link
```

- [ ] **Step 2: Commit**

```bash
git add presentation/slides_outline.md
git commit -m "docs: add presentation slides outline with content"
```

---

## Phase 4 Verification

- [ ] Report covers all 12 required sections
- [ ] Presentation outline covers all 6 required topics
- [ ] Report references actual metrics from model_metadata.json
- [ ] Report includes COVID-19 placement decision
- [ ] Report includes causal interpretation disclaimer

---

# Final Verification Checklist

| AGENTS.md Rule | Status |
|----------------|--------|
| 1. Verify code | Each task ends with pytest + verification |
| 2. Tests for every src/ file | test_config, test_data, test_features, test_train, test_evaluate, test_visualization |
| 3. Docstrings + type hints | All files have module docstrings, function docstrings, type hints |
| 4. Logging, no print() | Zero print() statements; all output via logging |
| 5. No silent failures | All error paths raise or log |
| 6. No leakage | Temporal split enforced; imputer in pipeline; log transform in pipeline; coverage on train only |
| 7. Lean notebooks | Markdown → code → output → interpretation pattern |
| 8. Deliberate visuals | Project palette defined and used consistently |
| 9. Minimal dependencies | No plotly (removed); no xgboost; only essential packages |
| 10. Follow spec | Streamlit only; no new frameworks |
| 11. Smallest change | Each task modifies minimal files |
| 12. Final status | Every response: what changed, tests, result, VERIFIED/NOT VERIFIED |

---

# Issue Resolution Summary

| ID | Issue | Resolution |
|----|-------|------------|
| B1 | .gitignore blocks deployment | Removed models/ and data/processed/ from gitignore |
| B2 | Persistence baseline wrong | Changed to predict current year's growth |
| B3 | Train/serve feature skew | Log transform in pipeline; full feature list from metadata |
| B4 | Extrapolation warning unreachable | Slider range widened to data min/max; warning on P1-P99 |
| B5 | Metadata never written | Added Task 2.3 (finalize_model.py) |
| B6 | Test syntax errors | Fixed `config.RANDOM_STATE` and explicit_codes in tests |
| G1 | No report task | Added Task 4.1 |
| G2 | No presentation | Added Task 4.2 with slide outline |
| G3 | __main__ blocks missing | Added to data.py and features.py |
| G4 | pytest not in requirements | Added to requirements.txt |
| G5 | LICENSE missing | Added Task 0.3 |
| G6 | country_metadata.csv not written | Added to finalize_model.py |
| G7 | Bootstrap CI not invoked | Added to evaluation notebook |
| M1 | Encoding latin-1 | Changed to utf-8 |
| M2 | Data audit missing | Added Task 0.6 |
| M3 | Region config mismatch | Config says "Africa"; code uses config value |
| M4 | Coverage on full panel | Now computed on training rows only |
| M5 | Heatmap shows one column | Now averages across all features |
| M6 | Explore page crash | Added empty check |
| M7 | page_performance idempotency | Reads from metadata, not recomputing |
| M8 | Config.RANDOM_STATE casing | Renamed to random_state |
| M9 | COVID placement undocumented | Added paragraph in report |
| M10 | Streamlit Python version | Noted 3.11 in deployment checklist |

---

# Execution Summary

| Phase | Tasks | Estimated Effort |
|-------|-------|-----------------|
| Phase 0: Scaffolding | 7 tasks | ~45 min |
| Phase 1: Data Pipeline | 4 tasks | ~2 hours |
| Phase 2: Modeling | 4 tasks | ~2.5 hours |
| Phase 3: Application | 3 tasks | ~2 hours |
| Phase 4: Deliverables | 2 tasks | ~1.5 hours |
| **Total** | **20 tasks** | **~8.5 hours** |

This fits within 2 days with margin for deployment debugging.
