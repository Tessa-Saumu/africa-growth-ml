"""Configuration loading and validation for the Africa Growth Explorer.

Loads indicator definitions, model hyperparameters, and temporal split
boundaries from a YAML configuration file.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List
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
    hgb_max_iter: int = 200
    hgb_learning_rate: float = 0.05
    hgb_max_depth: int = 3
    # C5 FIX: explicit early-stopping controls ("auto" is inert below 10k samples)
    hgb_l2_regularization: float = 1.0
    hgb_early_stopping: bool = True
    hgb_validation_fraction: float = 0.15
    hgb_n_iter_no_change: int = 15
    # Task 2.1: compact expanding-window CV grid (spec section 9)
    cv_initial_train_end: int = 2010
    cv_val_window: int = 2
    ridge_alpha_grid: List[float] = field(
        default_factory=lambda: [1.0, 10.0, 100.0, 300.0, 1000.0, 3000.0])
    hgb_grid: Dict[str, List[Any]] = field(default_factory=lambda: {
        "max_depth": [2, 3],
        "learning_rate": [0.01, 0.03, 0.05],
        "max_iter": [100, 200],
    })


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

    model_raw = raw["model"]
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
        random_state=model_raw["random_state"],
        ridge_alpha=model_raw["ridge_alpha"],
        hgb_max_iter=model_raw["hgb_max_iter"],
        hgb_learning_rate=model_raw["hgb_learning_rate"],
        hgb_max_depth=model_raw["hgb_max_depth"],
        hgb_l2_regularization=model_raw.get("hgb_l2_regularization", 1.0),
        hgb_early_stopping=model_raw.get("hgb_early_stopping", True),
        hgb_validation_fraction=model_raw.get("hgb_validation_fraction", 0.15),
        hgb_n_iter_no_change=model_raw.get("hgb_n_iter_no_change", 15),
        cv_initial_train_end=model_raw.get("cv_initial_train_end", 2010),
        cv_val_window=model_raw.get("cv_val_window", 2),
        ridge_alpha_grid=model_raw.get(
            "ridge_alpha_grid", [1.0, 10.0, 100.0, 300.0, 1000.0, 3000.0]),
        hgb_grid=model_raw.get("hgb_grid", {
            "max_depth": [2, 3],
            "learning_rate": [0.01, 0.03, 0.05],
            "max_iter": [100, 200],
        }),
    )
    logger.info(
        "Loaded %d features, target=%s, %d African countries",
        len(config.features), config.target_code, len(config.african_countries)
    )
    return config