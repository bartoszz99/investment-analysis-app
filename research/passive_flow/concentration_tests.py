"""
Event studies for passive-flow / concentration mechanics.
Events A/B/C — condition → forward distribution (not IC-only).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.passive_flow.distribution_analysis import (
    analyze_event_distribution,
    compare_to_unconditional,
)


def _expanding_high(series: pd.Series, q: float = 0.80, min_periods: int = 60) -> pd.Series:
    lag = series.shift(1)
    return lag >= lag.expanding(min_periods=min_periods).quantile(q)


def _expanding_low(series: pd.Series, q: float = 0.20, min_periods: int = 60) -> pd.Series:
    lag = series.shift(1)
    return lag <= lag.expanding(min_periods=min_periods).quantile(q)


def event_a_narrow_rally(
    features: pd.DataFrame,
    etf_close: pd.Series,
    *,
    range_window: int = 60,
) -> pd.Series:
    """
    SPY/ETF near top of rolling range + breadth deteriorating + concentration rising.
    """
    lag_etf = etf_close.shift(1)
    roll_min = lag_etf.rolling(range_window, min_periods=range_window).min()
    roll_max = lag_etf.rolling(range_window, min_periods=range_window).max()
    range_pct = (lag_etf - roll_min) / (roll_max - roll_min).replace(0, np.nan)
    near_high = range_pct >= 0.90

    breadth = features["breadth_pct_above_sma20"]
    breadth_weak = breadth.diff(5) < 0
    conc_rising = features["concentration_ratio_top10"].diff(5) > 0

    return near_high & breadth_weak & conc_rising


def event_b_passive_dominance(features: pd.DataFrame) -> pd.Series:
    """ETF strong, equal-weight weak, mega-cap share elevated."""
    etf_mom = features["etf_return_lagged"].rolling(20, min_periods=10).sum()
    # equal_weight_divergence = etf_ret - ew_ret  =>  ew_mom ≈ etf_mom - divergence_mom
    div_mom = features["equal_weight_divergence"].rolling(20, min_periods=10).sum()
    ew_mom = etf_mom - div_mom
    mega_high = _expanding_high(features["mega_cap_return_share"], 0.75)

    etf_strong = etf_mom > etf_mom.expanding(60).quantile(0.60)
    ew_weak = ew_mom < ew_mom.expanding(60).quantile(0.40)
    return etf_strong & ew_weak & mega_high


def event_c_participation_recovery(features: pd.DataFrame) -> pd.Series:
    """Breadth thrust + concentration falling."""
    breadth = features["breadth_pct_above_sma20"]
    breadth_surge = breadth.diff(5) > breadth.diff(5).expanding(60).quantile(0.75)
    conc_falling = features["concentration_ratio_top10"].diff(5) < 0
    return breadth_surge & conc_falling


def run_event_study(
    event_mask: pd.Series,
    etf_close: pd.Series,
    event_name: str,
    etf: str,
) -> dict:
    dist = analyze_event_distribution(event_mask, etf_close)
    cmp = compare_to_unconditional(event_mask, etf_close, horizon=20)
    return {
        "event": event_name,
        "etf": etf,
        "n_events": int(event_mask.fillna(False).sum()),
        "distributions": dist,
        "vs_unconditional_20d": cmp,
    }


def run_all_events(
    features: pd.DataFrame,
    etf_close: pd.Series,
    etf: str,
) -> list[dict]:
    studies = [
        run_event_study(event_a_narrow_rally(features, etf_close), etf_close, "A_narrow_rally", etf),
        run_event_study(event_b_passive_dominance(features), etf_close, "B_passive_dominance", etf),
        run_event_study(
            event_c_participation_recovery(features), etf_close, "C_participation_recovery", etf
        ),
    ]
    return studies
