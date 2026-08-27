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
WDI_DATA_FILE = "WDICSV.csv"
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
    african_codes: List[str],
) -> pd.DataFrame:
    """Filter to African countries only using explicit ISO3 list.

    B7 FIX: Uses an explicit ISO3 list (from config/indicators.yaml) instead of
    substring-matching the Region column. "Middle East & North Africa" contains
    the substring "Africa" and would silently include Saudi Arabia, Iran, etc.

    Args:
        df: Raw WDI dataframe.
        african_codes: List of ISO3 codes for African countries.

    Returns:
        Filtered dataframe with African countries only.
    """
    # M12: no aggregate-code filtering needed here. Aggregates (SSF, AFE, AFW,
    # WLD, income groups) cannot appear: the config list contains only
    # sovereign ISO3 codes. See tests/test_data.py.

    mask = df["Country Code"].isin(african_codes)
    filtered = df[mask].copy()
    logger.info(
        "Filtered to %d rows (%d countries) from %d total rows using ISO3 list",
        len(filtered),
        filtered["Country Code"].nunique(),
        len(df),
    )
    # Cross-check: log any ISO3 in data that's in our list but metadata might disagree
    data_codes = set(filtered["Country Code"].unique())
    missing_from_data = set(african_codes) - data_codes
    if missing_from_data:
        logger.warning("ISO3 codes in config but not in data: %s", missing_from_data)
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


def check_duplicates(df: pd.DataFrame, key_cols: List[str]) -> pd.DataFrame:
    """Detect duplicate keys and surface conflicting values.

    Spec section 7 requires that conflicting duplicates be investigated, not
    silently collapsed. Exact duplicates are safe to drop; conflicting ones
    indicate a source-data problem and are logged at WARNING.

    Args:
        df: Long-format frame prior to pivoting.
        key_cols: Columns forming the uniqueness key,
            e.g. ["iso3", "year", "indicator_code"].

    Returns:
        Rows belonging to duplicated keys, empty if none.
    """
    dup_mask = df.duplicated(subset=key_cols, keep=False)
    dups = df[dup_mask]
    if dups.empty:
        logger.info("Duplicate check on %s: none found", key_cols)
        return dups

    exact = df.duplicated(keep=False) & dup_mask
    n_exact = int(exact.sum())
    conflicting = dups[~dups.index.isin(df[exact].index)]
    logger.warning(
        "Duplicate check on %s: %d duplicated rows (%d exact, %d conflicting)",
        key_cols, len(dups), n_exact, len(conflicting),
    )
    if not conflicting.empty:
        logger.warning(
            "Conflicting duplicate keys require investigation:\n%s",
            conflicting.head(20).to_string(),
        )
    return dups


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
        config = load_config()
        # B7 FIX: Use explicit ISO3 list, not region substring filter
        african_df = filter_african_countries(df, african_codes=config.african_countries)
        indicator_codes = [f.code for f in config.features]
        indicator_codes.append(config.target_code)
        filtered = filter_indicators(african_df, indicator_codes)
        year_cols = [str(y) for y in range(config.min_year, 2025)]
        long = reshape_wide_to_long(filtered, year_cols)
        # M6: spec section 7 - investigate conflicting duplicates rather than
        # silently dropping them.
        check_duplicates(long, ["iso3", "year", "indicator_code"])
        panel = pivot_to_country_year(long)
        numeric_cols = [c for c in panel.columns if c not in ["iso3", "country_name", "year"]]
        panel = clean_numeric(panel, numeric_cols)
        log_dataset_info(panel, "Final Panel")
        panel.to_parquet("data/processed/model_data.parquet", index=False)
        logger.info("Saved to data/processed/model_data.parquet")
    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        sys.exit(1)