"""Shared research utilities."""

from research.common.feature_neutralization import (
    cross_sectional_demean,
    neutralize_feature_panel,
    rolling_ols_residual,
    rolling_zscore,
)
from research.common.forward_returns import HORIZONS, forward_return
from research.common.signal_evaluation import evaluate_signal, spearman_ic

__all__ = [
    "HORIZONS",
    "forward_return",
    "spearman_ic",
    "evaluate_signal",
    "rolling_ols_residual",
    "rolling_zscore",
    "neutralize_feature_panel",
    "cross_sectional_demean",
]
