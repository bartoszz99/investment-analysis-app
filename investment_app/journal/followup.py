"""
Kontrola dziennika — zmiana ceny od analizy, zgodność kierunku z tezą.
Śledzenie jakości rozumowania, nie księgowość portfela.
"""

from __future__ import annotations

import yfinance as yf


def current_price(ticker: str) -> float | None:
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="5d", auto_adjust=True)
        if hist is None or hist.empty:
            return None
        return float(hist["Close"].iloc[-1])
    except Exception:
        return None


def enrich_entry(entry: dict) -> dict:
    out = dict(entry)
    px0 = entry.get("price_at_analysis")
    ticker = entry.get("ticker", "")
    px1 = current_price(ticker) if ticker else None

    out["current_price"] = px1
    if px0 and px1 and px0 > 0:
        out["pct_change_since_analysis"] = (px1 / px0 - 1.0) * 100.0
    else:
        out["pct_change_since_analysis"] = None

    out["directionally_correct"] = _directional_check(
        entry.get("decision"),
        entry.get("idea_type"),
        out.get("pct_change_since_analysis"),
    )
    return out


def _directional_check(
    decision: str | None,
    idea_type: str | None,
    pct_change: float | None,
) -> str | None:
    if pct_change is None:
        return None

    bullish_ideas = {"momentum", "breakout", "earnings", "macro"}
    idea = (idea_type or "momentum").lower()

    if decision == "IGNORE":
        if abs(pct_change) < 3:
            return "Neutralnie — niskie przekonanie, cena spokojna."
        if pct_change > 5:
            return "Przegapiony ruch — odrzuciłeś, a cena ruszyła w górę (wróć do rozumowania)."
        return "Uniknięty szum — duży spadek lub bok."

    expected_up = idea in bullish_ideas or decision == "BUY"

    if expected_up and pct_change > 2:
        return "Zgodnie z kierunkiem — cena poszła z byczą tezą."
    if expected_up and pct_change < -2:
        return "Niezgodnie z kierunkiem — bycza teza, a cena spadła."
    if not expected_up and pct_change < 0:
        return "Zgodnie z kierunkiem — ostrożna postawa i słabsza cena."
    return "Mieszane — ruch ceny nie potwierdza ani nie obala tezy wprost."
