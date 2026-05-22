"""
Poland (GPW) structural diagnostics — practical investing, not quant alpha.

Focus: liquidity, speculation, narrative dependence, ownership flags.
No IC, factor neutralization, or pseudo-alpha metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from investment_app.data.ticker_mapper import ownership_flags, strip_suffix
from investment_app.i18n import risk_level_label, translate_flag


def _clip(x: float) -> float:
    return float(np.clip(x, -1.0, 1.0))


def _liquidity_quality(close: pd.Series, volume: pd.Series) -> dict:
    lag_v = volume.shift(1)
    lag_c = close.shift(1)
    turnover = (lag_v * lag_c).tail(60)
    avg_vol = lag_v.tail(60).mean()
    avg_turn = turnover.mean()
    missing_days = int((lag_v.tail(60) == 0).sum())
    vol_std = lag_v.tail(60).std()
    vol_mean = lag_v.tail(60).mean()
    stability = 1.0 - min(vol_std / vol_mean, 1.0) if vol_mean > 0 else 0.0

    if avg_turn < 2_000_000 or avg_vol < 100_000 or missing_days > 5:
        level = "HIGH"
        score = -0.7
    elif avg_turn < 8_000_000 or avg_vol < 300_000:
        level = "MEDIUM"
        score = -0.2
    else:
        level = "LOW"
        score = 0.4

    return {
        "liquidity_risk": level,
        "avg_daily_volume": float(avg_vol) if avg_vol == avg_vol else None,
        "avg_turnover_pln": float(avg_turn) if avg_turn == avg_turn else None,
        "missing_volume_days_60d": missing_days,
        "liquidity_stability": round(float(stability), 2),
        "liquidity_score": score,
        "summary": (
            f"Ryzyko płynności: {risk_level_label(level)}. "
            f"Śr. dzienny wolumen ~{avg_vol:,.0f} akcji; "
            f"stabilność obrotu {stability:.0%}."
            if avg_vol == avg_vol
            else "Ograniczone dane o płynności."
        ),
    }


def _speculation_risk(close: pd.Series, volume: pd.Series) -> dict:
    lag_c = close.shift(1)
    ret = lag_c.pct_change()
    vol20 = ret.rolling(20, min_periods=20).std()
    vol_z = volume.shift(1) / volume.shift(1).rolling(20, min_periods=20).mean() - 1.0
    mom20 = lag_c / lag_c.shift(20) - 1.0

    last_vol = vol20.iloc[-1] if len(vol20) else np.nan
    last_vz = vol_z.iloc[-1] if len(vol_z) else np.nan
    last_mom = mom20.iloc[-1] if len(mom20) else np.nan

    flags = 0
    if last_vol == last_vol and last_vol > 0.045:
        flags += 1
    if last_vz == last_vz and last_vz > 1.5:
        flags += 1
    if last_mom == last_mom and abs(last_mom) > 0.25:
        flags += 1
    # parabolic: 20d move > 30% with rising vol
    if last_mom == last_mom and last_mom > 0.30 and last_vz == last_vz and last_vz > 0.8:
        flags += 2

    if flags >= 3:
        level = "HIGH"
        score = -0.75
    elif flags >= 1:
        level = "MODERATE"
        score = -0.35
    else:
        level = "LOW"
        score = 0.25

    return {
        "speculation_risk": level,
        "volatility_20d": float(last_vol) if last_vol == last_vol else None,
        "volume_vs_avg": float(last_vz) if last_vz == last_vz else None,
        "return_20d": float(last_mom) if last_mom == last_mom else None,
        "speculation_score": score,
        "summary": (
            f"Ryzyko spekulacji: {risk_level_label(level)}. "
            + (
                "Ostatni ruch wygląda na napędzany wolumenem i rozciągnięty — "
                "typowe przy rajdach detalu na GPW."
                if level != "LOW"
                else "Brak skrajnego wzorca spekulacji krótkoterminowej."
            )
        ),
    }


def _narrative_dependence(
    close: pd.Series,
    fundamental_score: float,
    speculation: dict,
) -> dict:
    lag = close.shift(1)
    mom60 = (lag / lag.shift(60) - 1.0).iloc[-1] if len(lag) > 60 else np.nan
    spec_level = speculation.get("speculation_risk", "LOW")
    weak_fund = fundamental_score < 0.1

    score_penalty = 0
    if spec_level == "HIGH":
        score_penalty += 2
    elif spec_level == "MODERATE":
        score_penalty += 1
    if mom60 == mom60 and abs(mom60) > 0.35 and weak_fund:
        score_penalty += 2
    if mom60 == mom60 and abs(mom60) > 0.20 and weak_fund:
        score_penalty += 1

    if score_penalty >= 3:
        level = "HIGH"
        score = -0.7
    elif score_penalty >= 1:
        level = "MEDIUM"
        score = -0.3
    else:
        level = "LOW"
        score = 0.2

    return {
        "narrative_dependence": level,
        "narrative_score": score,
        "summary": (
            "Ostatni rajd może zależeć od historii i udziału detalu "
            "bardziej niż od widocznej poprawy biznesu."
            if level == "HIGH"
            else (
                "Pewne ryzyko narracji — cena częściowo wyprzedza fundamenty."
                if level == "MEDIUM"
                else "Zachowanie ceny nie jest wyraźnie oderwane od fundamentów."
            )
        ),
    }


def analyze_poland_structural(
    close: pd.Series,
    volume: pd.Series,
    *,
    ticker: str,
    fundamental_score: float = 0.0,
) -> dict:
    liq = _liquidity_quality(close, volume)
    spec = _speculation_risk(close, volume)
    narr = _narrative_dependence(close, fundamental_score, spec)
    flags = ownership_flags(ticker)
    extra = []
    base = strip_suffix(ticker)
    if base in ("PKN", "PKO", "PZU", "KGH"):
        extra.append("cyclical / commodity or rates sensitivity")


    composite = (
        0.35 * liq["liquidity_score"]
        + 0.35 * spec["speculation_score"]
        + 0.30 * narr["narrative_score"]
    )
    composite = _clip(composite)

    # Classification for decision layer
    if liq["liquidity_risk"] == "HIGH" or spec["speculation_risk"] == "HIGH":
        classification = "FRAGILE"
    elif narr["narrative_dependence"] == "HIGH":
        classification = "NARRATIVE_DRIVEN"
    elif liq["liquidity_risk"] == "MEDIUM" or spec["speculation_risk"] == "MODERATE":
        classification = "STRUCTURAL_HYPOTHESIS"
    else:
        classification = "ACCEPTABLE"

    summary_parts = [liq["summary"], spec["summary"], narr["summary"]]
    if flags:
        pl_flags = [translate_flag(f) for f in flags + extra]
        summary_parts.append("Flagi właścicielskie: " + "; ".join(pl_flags))

    return {
        "score_structural": composite,
        "classification": classification,
        "regime_stability": 0.7 if spec["speculation_risk"] == "LOW" else 0.4,
        "liquidity": liq,
        "speculation": spec,
        "narrative": narr,
        "ownership_flags": flags + extra,
        "summary": " ".join(summary_parts),
        "market": "POLAND",
    }
