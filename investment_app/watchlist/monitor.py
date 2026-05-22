"""
Ponowna analiza pozycji z listy obserwowanych i wykrywanie zmian wyniku/decyzji.
"""

from __future__ import annotations

from investment_app.engine import AnalysisRequest, analyze
from investment_app.i18n import decision_label, translate
from investment_app.watchlist.store import list_items, update_item_snapshot


def _risk_flags(result) -> list[str]:
    flags = []
    if result.structural.get("classification") == "MARKET_EXPOSURE":
        flags.append("Dominuje ekspozycja rynkowa")
    if result.structural.get("classification") == "NOISE":
        flags.append("Słabe wsparcie strukturalne")
    if result.structural.get("regime_stability", 1) < 0.5:
        flags.append("Niestabilne między reżimami rynku")
    if result.technical.get("vol_regime") == "high":
        flags.append("Wysoka zmienność")
    return flags


def refresh_item(item: dict, *, save_journal: bool = False) -> dict:
    req = AnalysisRequest(
        ticker=item["ticker"],
        asset_type=item.get("asset_type", "Stock"),
        idea_type=item.get("idea_type", "momentum"),
        horizon=item.get("horizon", "medium"),
        market_region=item.get("market_region", "USA"),
    )
    result = analyze(req, save_journal=save_journal)
    snapshot = {
        "last_score": result.final_score,
        "last_decision": result.decision,
        "last_structural": result.structural.get("classification"),
        "last_trend": result.technical.get("trend"),
        "risk_flags": _risk_flags(result),
    }
    update_item_snapshot(item["ticker"], snapshot)

    prev_dec = item.get("last_decision") or item.get("prev_decision")
    prev_score = item.get("last_score")
    if prev_score is not None:
        snapshot["score_delta"] = result.final_score - prev_score
    else:
        snapshot["score_delta"] = None

    snapshot["alerts"] = _alerts(item, result, prev_dec, prev_score)
    snapshot["ticker"] = result.ticker
    return snapshot | {"result": result}


def _alerts(item: dict, result, prev_dec, prev_score) -> list[str]:
    alerts = []
    if prev_score is not None:
        delta = result.final_score - prev_score
        if delta <= -0.15:
            alerts.append("Wynik mocno spadł — jakość strukturalna może się pogarszać.")
        elif delta >= 0.15:
            alerts.append("Wynik się poprawił — pomysł zyskuje spójność.")

    if prev_dec == "WATCH" and result.decision == "BUY":
        alerts.append(
            f"Awans: {decision_label('WATCH')} → {decision_label('BUY')}"
        )
    if prev_dec == "BUY" and result.decision == "WATCH":
        alerts.append(
            f"Spadek: {decision_label('BUY')} → {decision_label('WATCH')}"
        )
    if prev_dec == "BUY" and result.decision == "IGNORE":
        alerts.append(
            f"Spadek: {decision_label('BUY')} → {decision_label('IGNORE')} — wróć do tezy"
        )

    if result.structural.get("classification") == "MARKET_EXPOSURE":
        alerts.append("Teraz sklasyfikowane jako ekspozycja rynkowa")
    return alerts


def refresh_all(*, save_journal: bool = False) -> list[dict]:
    rows = []
    for item in list_items():
        try:
            rows.append(refresh_item(item, save_journal=save_journal))
        except Exception as e:
            rows.append({
                "ticker": item["ticker"],
                "error": str(e),
                "alerts": [translate("page.watchlist.refresh_failed", error=str(e))],
            })
    return rows
