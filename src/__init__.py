"""NBA Prediction Market — built on real NBA.com game data (2018-2024)."""

from .nba_real_pipeline import NBARealDataPipeline
from .models import EnsemblePredictor, LogisticModel, GradientBoostModel, NeuralNetModel
from .optimization import KellySweep, PortfolioKellyOptimizer, GrowthOptimizer, BetCorrelationAnalyzer
from .utils import (
    kelly_criterion,
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    profit_factor,
    calibration_error,
    implied_probability,
    expected_value
)

__version__ = "0.2.0"
__author__ = "Victoria"

__all__ = [
    "NBARealDataPipeline",
    "EnsemblePredictor",
    "LogisticModel",
    "GradientBoostModel",
    "NeuralNetModel",
    "KellySweep",
    "PortfolioKellyOptimizer",
    "GrowthOptimizer",
    "BetCorrelationAnalyzer",
    "kelly_criterion",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "profit_factor",
    "calibration_error",
    "implied_probability",
    "expected_value"
]
