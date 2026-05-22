"""
Oś fundamentalna — lekka jakość biznesu / wycena (nie trend ceny).
Wynik ∈ [-1, 1].
"""

from __future__ import annotations

import numpy as np


def _clip(x: float) -> float:
    return float(np.clip(x, -1.0, 1.0))


def analyze_fundamental(ticker: str, info: dict | None) -> dict:
    if not info:
        return {
            "score_fundamental": 0.0,
            "summary": "Ograniczone dane fundamentalne — wynik neutralny.",
            "metrics": {},
        }

    pe = info.get("trailingPE") or info.get("forwardPE")
    rev_g = info.get("revenueGrowth")
    margin = info.get("profitMargins") or info.get("operatingMargins")
    debt_eq = info.get("debtToEquity")
    earn_trend = info.get("earningsGrowth") or info.get("earningsQuarterlyGrowth")

    parts: list[float] = []
    notes: list[str] = []

    if rev_g is not None and rev_g == rev_g:
        parts.append(_clip(rev_g * 2.0))
        notes.append(f"Wzrost przychodów: {rev_g:.0%}")

    if margin is not None and margin == margin:
        parts.append(_clip((margin - 0.10) * 3.0))
        notes.append(f"Marża zysku: {margin:.0%}")

    if earn_trend is not None and earn_trend == earn_trend:
        parts.append(_clip(earn_trend * 1.5))
        notes.append(f"Trend zysków: {earn_trend:.0%}")

    if pe is not None and pe == pe and pe > 0:
        if pe < 8:
            parts.append(-0.15)
            notes.append(f"C/Z {pe:.0f} — możliwy dystres lub dołek cyklu")
        elif pe > 45:
            parts.append(-0.25)
            notes.append(f"C/Z {pe:.0f} — wysokie oczekiwania")
        else:
            parts.append(0.1)
            notes.append(f"C/Z {pe:.0f} — w typowym zakresie")

    if debt_eq is not None and debt_eq == debt_eq:
        if debt_eq > 200:
            parts.append(-0.3)
            notes.append(f"Dług/kapitał {debt_eq:.0f} — wysoka dźwignia")
        elif debt_eq < 80:
            parts.append(0.1)
            notes.append(f"Dług/kapitał {debt_eq:.0f} — umiarkowana dźwignia")

    score = float(np.mean(parts)) if parts else 0.0
    score = _clip(score)

    if score > 0.25:
        quality = "Fundamenty biznesu wyglądają wspierająco."
    elif score < -0.25:
        quality = "Profil fundamentalny budzi ostrożność."
    else:
        quality = "Fundamenty są mieszane lub przeciętne."

    return {
        "score_fundamental": score,
        "summary": quality,
        "detail": " ".join(notes[:4]) if notes else "Brak szczegółowych metryk.",
        "metrics": {
            "pe": pe,
            "revenue_growth": rev_g,
            "margin": margin,
            "debt_to_equity": debt_eq,
            "earnings_trend": earn_trend,
        },
    }
