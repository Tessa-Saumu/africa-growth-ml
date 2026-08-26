"""Feature engineering, target creation, and temporal splitting.

Builds the country-year panel, creates the next-year GDP growth target,
applies feature selection by coverage (computed on training data only),
and produces train/val/test splits.
"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


# B8 FIX: Named function in src/ so it's picklable by module path.
# Never use a lambda or notebook-local function - Streamlit Cloud
# needs to import it by module path when loading the serialized pipeline.
def clip_log1p(X: np.ndarray) -> np.ndarray:
    """Apply log1p transform after clipping negatives to 0.

    Used inside sklearn pipelines via FunctionTransformer. Legitimate negative
    values exist in WDI data (inflation, FDI can be negative); log1p of
    anything < -1 produces NaN.

    Args:
        X: Input array (may contain negative values).

    Returns:
        log1p-transformed array with negatives clipped to 0.
    """
    return np.log1p(np.clip(X, 0, None))


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