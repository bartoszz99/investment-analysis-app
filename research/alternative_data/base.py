"""
Alternative data research — shared types and lag utilities.
Does NOT feed production signals, portfolio, or execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd


class LeakageRisk(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass
class SignalMetadata:
    name: str
    source: str
    lag_days: int
    update_frequency: str
    leakage_risk: LeakageRisk
    timestamp_assumption: str
    lag_policy: str
    description: str = ""


@dataclass
class AlternativeSignal:
    metadata: SignalMetadata
    series: pd.Series
    stats: dict[str, float] = field(default_factory=dict)

    def with_stats(self) -> AlternativeSignal:
        s = self.series.dropna()
        if s.empty:
            self.stats = {"mean": np.nan, "std": np.nan, "n_obs": 0}
        else:
            self.stats = {
                "mean": float(s.mean()),
                "std": float(s.std()),
                "min": float(s.min()),
                "max": float(s.max()),
                "n_obs": int(len(s)),
            }
        return self


def apply_lag(series: pd.Series, lag_days: int, reason: str = "") -> pd.Series:
    """
    Shift signal forward so value at t uses only info through t-lag.
    lag_days=1: signal[t] known before Open[t] (published prior close).
    """
    if lag_days < 1:
        raise ValueError("lag_days must be >= 1 for causal research signals")
    return series.shift(lag_days)


def normalize_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Strip timezone and normalize to midnight for calendar alignment."""
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return pd.DatetimeIndex([pd.Timestamp(d).normalize() for d in idx])


def align_to_calendar(series: pd.Series, calendar: pd.DatetimeIndex) -> pd.Series:
    """Reindex to trading calendar (date-normalized keys)."""
    if series.empty:
        return pd.Series(np.nan, index=calendar)
    s = series.copy()
    s.index = normalize_index(s.index)
    cal = normalize_index(calendar)
    return s.reindex(cal)


def expanding_zscore(series: pd.Series, min_periods: int = 20) -> pd.Series:
    """Causal z-score: uses expanding history only (no full-sample stats)."""
    lagged = series.shift(1)
    mu = lagged.expanding(min_periods=min_periods).mean()
    sigma = lagged.expanding(min_periods=min_periods).std()
    return (lagged - mu) / sigma.replace(0, np.nan)
