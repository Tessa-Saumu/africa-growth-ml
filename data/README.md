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