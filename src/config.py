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
    african_countries: List[str]
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
        african_countries=raw["geographic"]["african_countries"],
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
        "Loaded %d features, target=%s, %d African countries",
        len(config.features), config.target_code, len(config.african_countries)
    )
    return config