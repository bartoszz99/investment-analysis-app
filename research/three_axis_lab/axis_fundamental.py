"""
Fundamental axis — business quality / style (not price-based trend).
Score ∈ [-1, 1]. Uses yfinance info when available; sparse data → neutral 0.
"""

from __future__ import annotations

import numpy as np


def _clip_score(x: float) -> float:
    return float(np.clip(x, -1.0, 1.0))


def score_fundamental(ticker: str, info: dict | None) -> dict:
    """
    Heuristic quality/style score — diagnostic only, not a valuation model.

    + growth proxy (revenue growth, margins)
    + value sanity (P/E not extreme)
    + sector context label
    """
    if not info:
        return {
            "score_fundamental": 0.0,
            "quality_trend": "unknown",
            "sector": "unknown",
            "style_proxy": "unknown",
            "notes": "no fundamental data",
        }

    pe = info.get("trailingPE") or info.get("forwardPE")
    rev_g = info.get("revenueGrowth")
    margin = info.get("profitMargins") or info.get("operatingMargins")
    sector = info.get("sector") or info.get("industry") or "unknown"

    components: list[float] = []

    # Revenue growth → quality improvement proxy
    if rev_g is not None and rev_g == rev_g:
        components.append(_clip_score(rev_g * 2.0))

    # Margin level (not price)
    if margin is not None and margin == margin:
        components.append(_clip_score((margin - 0.10) * 3.0))

    # P/E sanity: extreme values penalized (bubble / distress)
    if pe is not None and pe == pe and pe > 0:
        if pe < 8:
            components.append(-0.2)
        elif pe > 45:
            components.append(-0.4)
        else:
            components.append(0.15)

    score = float(np.mean(components)) if components else 0.0
    score = _clip_score(score)

    if rev_g is not None and rev_g == rev_g:
        if rev_g > 0.08:
            quality = "improving"
        elif rev_g < -0.02:
            quality = "deteriorating"
        else:
            quality = "stable"
    else:
        quality = "unknown"

    pe_val = pe if pe and pe == pe else None
    if pe_val and pe_val > 30:
        style = "growth"
    elif pe_val and pe_val < 15:
        style = "value"
    else:
        style = "blend"

    return {
        "score_fundamental": score,
        "quality_trend": quality,
        "sector": sector,
        "style_proxy": style,
        "pe": pe,
        "revenue_growth": rev_g,
        "margin_proxy": margin,
    }
