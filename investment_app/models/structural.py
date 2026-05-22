"""
Oś strukturalna — niezależna struktura czy głównie ekspozycja rynkowa?

Kontrole szeregu czasowego (nie przekrój badawczy).
Neutralizacja tylko wsteczna; bez dopasowania na pełnej próbce.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


IC_MIN = 0.05
COLLAPSE = 0.5


def _spearman(a: pd.Series, b: pd.Series) -> float:
    aligned = pd.concat([a, b], axis=1).dropna()
    if len(aligned) < 40:
        return np.nan
    return float(aligned.iloc[:, 0].rank().corr(aligned.iloc[:, 1].rank()))


def _build_signal(close: pd.Series, idea: str) -> pd.Series:
    lag = close.shift(1)
    idea = idea.lower()
    if idea == "breakout":
        hi = lag.rolling(20, min_periods=20).max()
        return (lag / hi - 1.0).shift(1)
    if idea in ("value", "mean_reversion"):
        mu = lag.rolling(20, min_periods=20).mean()
        return (-(lag / mu - 1.0)).shift(1)
    if idea == "earnings":
        return (close / close.shift(1) - 1.0).shift(1)
    return (lag / lag.shift(126) - 1.0).shift(1)


def _forward_return(close: pd.Series, h: int) -> pd.Series:
    return close.shift(-h) / close - 1.0


def _neutralize(signal: pd.Series, spy_ret: pd.Series, window: int = 60) -> pd.Series:
    mp = max(20, window // 3)
    x, y = spy_ret.shift(1), signal.shift(1)
    mean_x = x.rolling(window, min_periods=mp).mean()
    mean_y = y.rolling(window, min_periods=mp).mean()
    cov = (x * y).rolling(window, min_periods=mp).mean() - mean_x * mean_y
    var = x.rolling(window, min_periods=mp).var()
    beta = cov / var.replace(0, np.nan)
    fitted = beta * x
    return signal - fitted


def _regime_ic(signal: pd.Series, fwd: pd.Series, mask: pd.Series) -> float:
    m = mask.reindex(signal.index).fillna(False)
    return _spearman(signal[m], fwd[m])


def analyze_structural(
    close: pd.Series,
    spy_close: pd.Series,
    idea: str,
) -> dict:
    signal = _build_signal(close, idea)
    spy_ret = spy_close.reindex(close.index).ffill().pct_change()
    residual = _neutralize(signal, spy_ret)
    fwd5 = _forward_return(close, 5)

    ic_raw = _spearman(signal, fwd5)
    ic_res = _spearman(residual, fwd5)

    lag_spy = spy_close.shift(1)
    ma200 = lag_spy.rolling(200, min_periods=100).mean()
    bull = lag_spy > ma200
    ic_bull = _regime_ic(signal, fwd5, bull)
    ic_bear = _regime_ic(signal, fwd5, ~bull)
    unstable = (
        ic_bull == ic_bull
        and ic_bear == ic_bear
        and abs(ic_bull) > 0.03
        and abs(ic_bear) > 0.03
        and ic_bull * ic_bear < 0
    )

    spy_corr = close.pct_change().rolling(60, min_periods=30).corr(spy_close.pct_change())
    last_corr = float(spy_corr.iloc[-1]) if len(spy_corr) else np.nan

    if ic_raw != ic_raw or abs(ic_raw) < IC_MIN:
        label = "NOISE"
        drivers = "Sygnał ma słaby związek z krótkoterminowymi zwrotami."
    elif unstable:
        label = "NOISE"
        drivers = "Efekt zmienia znak między hossą a bessą — prawdopodobnie niestabilny."
    elif ic_res == ic_res and abs(ic_res) < abs(ic_raw) * COLLAPSE:
        label = "MARKET_EXPOSURE"
        drivers = "Po usunięciu współruchu z rynkiem sygnał znika — beta / dryf sektora."
    elif ic_res == ic_res and abs(ic_res) >= IC_MIN:
        label = "TRUE_SIGNAL"
        drivers = "Część struktury zostaje po neutralizacji rynku — to nie gwarancja alfy."
    else:
        label = "STRUCTURAL_HYPOTHESIS"
        drivers = "Słaba, ale nie trywialna struktura — traktuj jako hipotezę, nie dowód."

    base = ic_res if ic_res == ic_res else ic_raw
    if label == "MARKET_EXPOSURE":
        score = np.sign(base) * min(abs(ic_raw or 0) / 0.15, 0.4) if base == base else 0.0
    elif label == "NOISE":
        score = 0.0
    else:
        score = np.sign(base) * min(abs(base) / 0.12, 1.0) if base == base else 0.0

    regime_stability = 0.35 if unstable else 0.85

    return {
        "score_structural": float(np.clip(score, -1, 1)),
        "classification": label,
        "ic_raw": ic_raw,
        "ic_residual": ic_res,
        "regime_stability": regime_stability,
        "spy_correlation": last_corr,
        "drivers": drivers,
        "summary": drivers,
    }
