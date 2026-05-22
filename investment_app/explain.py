"""
Wyjaśnienia po polsku — tryb USA i Polska (GPW).
"""

from __future__ import annotations

from investment_app.data.market_region import is_poland

_COUNTERPARTY_USA = {
    "momentum": (
        "Kto jest po drugiej stronie? Często fundusze trendu, przepływy ETF i CTA.",
        "Co zwykle to napędza? Ruch szerokiego rynku i rotacja sektorowa — nie ukryta przewaga.",
    ),
    "breakout": (
        "Kto jest po drugiej stronie? Klastry stop-lossów i zabezpieczenia short-gamma.",
        "Co zwykle to napędza? Presja pozycji krótkoterminowej — wybicia często zawodzą.",
    ),
    "earnings": (
        "Kto jest po drugiej stronie? Szybcy traderzy nagłówków; instytucje dostosowują się wolniej.",
        "Co zwykle to napędza? Przewartościowanie po wynikach — timing ma znaczenie.",
    ),
    "value": (
        "Kto jest po drugiej stronie? Fundusze czynnikowe i kontrarianie balansujący styl.",
        "Co zwykle to napędza? Ekspozycja na styl — wartość może odstawać latami.",
    ),
    "macro": (
        "Kto jest po drugiej stronie? Pasywni alokatorzy i rebalans emerytur.",
        "Co zwykle to napędza? Szerokie ruchy risk-on / risk-off.",
    ),
}

_COUNTERPARTY_PL = {
    "momentum": (
        "Kto jest po drugiej stronie? Często lokalny detal i traderzy tematyczni na GPW.",
        "Co zwykle to napędza? Historie i przepływy — na GPW cena bywa przed fundamentami.",
    ),
    "breakout": (
        "Kto jest po drugiej stronie? Traderzy krótkoterminowi goniący nagłówki i wolumen.",
        "Co zwykle to napędza? Cienka płynność potęguje wybicia i odwrócenia.",
    ),
    "earnings": (
        "Kto jest po drugiej stronie? Lokalne fundusze i detal reagujący na polskie raporty.",
        "Co zwykle to napędza? Luka oczekiwań — sprawdź płynność przed działaniem.",
    ),
    "value": (
        "Kto jest po drugiej stronie? Lokalni value i łowcy dywidendy.",
        "Co zwykle to napędza? Przewartościowanie cykliczne — polityka i PLN nadal liczą się.",
    ),
    "macro": (
        "Kto jest po drugiej stronie? Krajowe instytucje przy stopach i PLN.",
        "Co zwykle to napędza? Makro na GPW często dominuje nad historią pojedynczej spółki.",
    ),
}

GPW_RISK_HINTS = [
    "Udział państwa lub wrażliwość na politykę",
    "Niski free float — cena może skoczyć przy małym wolumenie",
    "Spekulacja i rajdy narracyjne napędzane detalem",
    "Ekspozycja na surowce / stopy procentowe",
    "Nagłówki polityczne i regulacyjne",
]


def build_explanation(
    ticker: str,
    idea: str,
    horizon: str,
    asset_type: str,
    fundamental: dict,
    technical: dict,
    structural: dict,
    decision: dict,
    *,
    market_region: str = "USA",
) -> dict:
    if is_poland(market_region):
        return _build_poland_explanation(
            ticker, idea, horizon, asset_type, fundamental, technical, structural, decision
        )
    return _build_usa_explanation(
        ticker, idea, horizon, asset_type, fundamental, technical, structural, decision
    )


def _build_usa_explanation(
    ticker: str,
    idea: str,
    horizon: str,
    asset_type: str,
    fundamental: dict,
    technical: dict,
    structural: dict,
    decision: dict,
) -> dict:
    cp = _COUNTERPARTY_USA.get(idea.lower(), _COUNTERPARTY_USA["momentum"])
    dec = decision["decision"]
    headline = _headline(dec)
    risks = _risks_usa(structural, technical, fundamental)
    driving = _what_drives_usa(structural, technical, idea)

    return {
        "headline": headline,
        "market_region": "USA",
        "fundamental_blurb": fundamental.get("summary", ""),
        "technical_blurb": technical.get("summary", ""),
        "structural_blurb": structural.get("summary", ""),
        "counterparty": cp[0],
        "likely_driver": cp[1],
        "what_is_driving_this": driving,
        "key_risks": risks,
        "structural_class": structural.get("classification"),
        "gpw_warnings": [],
    }


def _build_poland_explanation(
    ticker: str,
    idea: str,
    horizon: str,
    asset_type: str,
    fundamental: dict,
    technical: dict,
    structural: dict,
    decision: dict,
) -> dict:
    cp = _COUNTERPARTY_PL.get(idea.lower(), _COUNTERPARTY_PL["momentum"])
    dec = decision["decision"]
    liq = structural.get("liquidity", {})
    spec = structural.get("speculation", {})
    narr = structural.get("narrative", {})
    flags = structural.get("ownership_flags", [])

    headline = _headline_pl(dec, liq, spec, narr)
    risks = _risks_poland(structural, technical, fundamental, flags)
    driving = _what_drives_poland(structural, technical, idea, spec, narr)

    gpw_warnings = []
    if liq.get("liquidity_risk") == "HIGH":
        gpw_warnings.append("Płynność jest cienka — wyjście może być kosztowne.")
    if spec.get("speculation_risk") in ("HIGH", "MODERATE"):
        gpw_warnings.append("Ostatni rajd może być napędzany wolumenem i spekulacją.")
    if narr.get("narrative_dependence") in ("HIGH", "MEDIUM"):
        gpw_warnings.append("Cena może zależeć od narracji bardziej niż od fundamentów.")

    return {
        "headline": headline,
        "market_region": "POLAND",
        "fundamental_blurb": fundamental.get("summary", ""),
        "technical_blurb": technical.get("summary", ""),
        "structural_blurb": structural.get("summary", ""),
        "counterparty": cp[0],
        "likely_driver": cp[1],
        "what_is_driving_this": driving,
        "key_risks": risks,
        "structural_class": structural.get("classification"),
        "gpw_warnings": gpw_warnings,
        "gpw_diagnostics": {
            "liquidity_risk": liq.get("liquidity_risk"),
            "speculation_risk": spec.get("speculation_risk"),
            "narrative_dependence": narr.get("narrative_dependence"),
            "ownership_flags": flags,
        },
        "gpw_structural_risks": GPW_RISK_HINTS + flags,
    }


def _headline(dec: str) -> str:
    if dec == "BUY":
        return "Teza wygląda spójnie — to nie gwarancja zysku."
    if dec == "WATCH":
        return "Ciekawy pomysł, ale struktura lub timing nie są w pełni przekonujące."
    return "Słaba lub głównie ekspozycja rynkowa — niski priorytet."


def _headline_pl(dec: str, liq: dict, spec: dict, narr: dict) -> str:
    if liq.get("liquidity_risk") == "HIGH" or spec.get("speculation_risk") == "HIGH":
        return "Strukturalnie kruche na GPW — dyscyplina przed przekonaniem."
    if dec == "BUY":
        return "Rozumowanie może się bronić — sprawdź płynność i ryzyko narracji na GPW."
    if dec == "WATCH":
        return "Warto obserwować — narracja lub płynność mogą dominować."
    return "Niski priorytet — spekulacja, płynność lub słaby case biznesowy."


def _risks_usa(structural: dict, technical: dict, fundamental: dict) -> list[str]:
    risks = []
    if structural.get("classification") == "MARKET_EXPOSURE":
        risks.append("Prawdopodobnie beta rynku lub sektora, a nie przewaga na spółce.")
    if structural.get("classification") == "NOISE":
        risks.append("Brak stabilnego związku między pomysłem a przyszłymi zwrotami.")
    if technical.get("vol_regime") == "high":
        risks.append("Podwyższona zmienność — szersze wahania i wyższy koszt realizacji.")
    if fundamental.get("score_fundamental", 0) < -0.25:
        risks.append("Tło fundamentalne jest słabe.")
    return risks or ["Standardowe ryzyko rynku — makro, wyniki, płynność."]


def _risks_poland(
    structural: dict,
    technical: dict,
    fundamental: dict,
    flags: list[str],
) -> list[str]:
    from investment_app.i18n import translate_flag

    risks = []
    liq = structural.get("liquidity", {})
    spec = structural.get("speculation", {})
    if liq.get("liquidity_risk") == "HIGH":
        risks.append("Niska płynność — trudno wejść/wyjść bez poruszenia ceny.")
    if spec.get("speculation_risk") == "HIGH":
        risks.append(
            "Ostatni rajd wygląda na napędzany wolumenem i może zależeć od udziału "
            "detalu bardziej niż od poprawy fundamentów."
        )
    if structural.get("narrative", {}).get("narrative_dependence") == "HIGH":
        risks.append("Narracja może wyprzedzać zweryfikowaną poprawę biznesu.")
    if technical.get("vol_regime") == "high":
        risks.append("Wysoka zmienność — typowa na GPW przy ruchach tematycznych.")
    if fundamental.get("score_fundamental", 0) < -0.15:
        risks.append("Słaby ekran jakości biznesu — historię może nieść cena.")
    for f in flags:
        risks.append(translate_flag(f) if isinstance(f, str) else f)
    return risks or list(GPW_RISK_HINTS[:3])


def _what_drives_usa(structural: dict, technical: dict, idea: str) -> str:
    cls = structural.get("classification", "")
    trend = technical.get("trend", "mixed")
    if cls == "MARKET_EXPOSURE":
        return f"Prawdopodobnie szeroki rynek USA lub sektor ({idea}), a nie niezależna przewaga."
    if trend == "uptrend":
        return f"Udział w trendzie — klasyczny setup {idea}, często zatłoczony w large cap."
    return "Mieszane siły — potwierdź fundamentami przed zwiększeniem przekonania."


def _what_drives_poland(
    structural: dict,
    technical: dict,
    idea: str,
    spec: dict,
    narr: dict,
) -> str:
    if spec.get("speculation_risk") == "HIGH":
        return (
            "Ostatni rajd wygląda na napędzany wolumenem i może zależeć od udziału "
            "detalu bardziej niż od poprawy fundamentów."
        )
    if narr.get("narrative_dependence") == "HIGH":
        return (
            "Tematyczna historia może poruszać spółkę bardziej niż raportowane wyniki "
            "lub poprawa bilansu."
        )
    trend = technical.get("trend", "mixed")
    if trend == "uptrend":
        return f"Trend wzrostowy na GPW — sprawdź, czy płynność wspiera tezę {idea}."
    return "Mieszane motory — na GPW zawsze oddziel historię od substancji."
