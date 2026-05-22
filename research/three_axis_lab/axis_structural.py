"""
Structural-mathematical axis — IC, neutralization, regime stability.
Classification: TRUE_SIGNAL | MARKET_EXPOSURE | NOISE
"""

from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd

from research.common.forward_returns import forward_return
from research.common.signal_evaluation import spearman_ic
from research.three_axis_lab.neutralization import build_factor_panel, neutralize_signal

IC_MIN = 0.05
COLLAPSE_RATIO = 0.5
HORIZONS = (1, 5, 20)


class StructuralClass(str, Enum):
    TRUE_SIGNAL = "TRUE_SIGNAL"
    MARKET_EXPOSURE = "MARKET_EXPOSURE"
    NOISE = "NOISE"


def build_idea_signal(close: pd.Series, idea: str) -> pd.Series:
    """Idea-specific signal (lagged)."""
    lag = close.shift(1)
    idea = idea.lower()
    if idea == "breakout":
        hi = lag.rolling(20, min_periods=20).max()
        return (lag / hi - 1.0).shift(1)
    if idea == "mean_reversion":
        mu = lag.rolling(20, min_periods=20).mean()
        return (-(lag / mu - 1.0)).shift(1)
    if idea == "earnings_reaction":
        gap = close / close.shift(1) - 1.0
        return gap.shift(1)
    # default momentum
    return (lag / lag.shift(126) - 1.0).shift(1)


def regime_masks(spy_close: pd.Series) -> dict[str, pd.Series]:
    lag = spy_close.shift(1)
    ma200 = lag.rolling(200, min_periods=100).mean()
    vol = lag.pct_change().rolling(20, min_periods=20).std()
    return {
        "bull": lag > ma200,
        "bear": lag <= ma200,
        "high_vol": vol > vol.expanding(60).median(),
        "low_vol": vol <= vol.expanding(60).median(),
    }


def score_structural(
    ticker: str,
    close: pd.Series,
    volume: pd.Series,
    *,
    idea: str,
    spy_close: pd.Series,
    sector_close: pd.Series | None,
) -> dict:
    signal = build_idea_signal(close, idea)
    spy_ret = spy_close.pct_change()
    sec_ret = sector_close.pct_change() if sector_close is not None else None
    mom_proxy = spy_close.shift(1) / spy_close.shift(1).shift(126) - 1.0

    factors = build_factor_panel(ticker, spy_ret, sec_ret, mom_proxy)
    residual = neutralize_signal(signal, factors)

    ics: dict[int, float] = {}
    ic_res: dict[int, float] = {}
    for h in HORIZONS:
        fwd = forward_return(close, h)
        ics[h] = spearman_ic(signal, fwd)
        ic_res[h] = spearman_ic(residual, fwd)

    ic_mean = float(np.nanmean(list(ics.values())))
    ic_res_mean = float(np.nanmean(list(ic_res.values())))

    # Regime IC (5d)
    fwd5 = forward_return(close, 5)
    masks = regime_masks(spy_close)
    regime_ic = {}
    for name, mask in masks.items():
        sub = signal[mask.reindex(signal.index).fillna(False)]
        regime_ic[name] = spearman_ic(sub, fwd5.reindex(sub.index))

    signs = [v for v in regime_ic.values() if v == v and abs(v) > 0.01]
    unstable = len(signs) >= 2 and min(s for s in signs) * max(s for s in signs) < 0

    # Classification
    if abs(ic_mean) < IC_MIN:
        struct_class = StructuralClass.NOISE
        reasons = [f"|IC| < {IC_MIN}"]
    elif unstable:
        struct_class = StructuralClass.NOISE
        reasons = ["regime IC sign unstable"]
    elif abs(ic_res_mean) < abs(ic_mean) * COLLAPSE_RATIO:
        struct_class = StructuralClass.MARKET_EXPOSURE
        reasons = ["IC collapses after SPY/sector/momentum neutralization"]
    elif ic_res_mean == ic_res_mean and abs(ic_res_mean) >= IC_MIN:
        struct_class = StructuralClass.TRUE_SIGNAL
        reasons = ["residual IC above threshold"]
    else:
        struct_class = StructuralClass.NOISE
        reasons = ["weak residual structure"]

    # Map to score [-1, 1]
    if struct_class == StructuralClass.TRUE_SIGNAL:
        score = np.sign(ic_res_mean) * min(abs(ic_res_mean) / 0.15, 1.0)
    elif struct_class == StructuralClass.MARKET_EXPOSURE:
        score = np.sign(ic_mean) * min(abs(ic_mean) / 0.15, 0.5)
    else:
        score = 0.0

    # Flow proxy: volume z-score level
    lv = volume.shift(1)
    vol_z = (lv - lv.rolling(20, min_periods=20).mean()) / lv.rolling(20, min_periods=20).std()
    spy_corr = close.pct_change().rolling(60, min_periods=30).corr(spy_close.pct_change())

    return {
        "score_structural": float(np.clip(score, -1, 1)),
        "structural_class": struct_class.value,
        "ic_1d": ics.get(1, np.nan),
        "ic_5d": ics.get(5, np.nan),
        "ic_20d": ics.get(20, np.nan),
        "ic_mean": ic_mean,
        "residual_ic_mean": ic_res_mean,
        "regime_ic": regime_ic,
        "spy_corr_last": float(spy_corr.iloc[-1]) if len(spy_corr) else np.nan,
        "volume_z_last": float(vol_z.iloc[-1]) if len(vol_z) else np.nan,
        "reasons": reasons,
    }
