"""

Teza vs analiza — dowody potwierdzające / sprzeczne (reguły, bez ML).

"""



from __future__ import annotations



from investment_app.explain import GPW_RISK_HINTS

from investment_app.i18n import classification_label, decision_label



_KEYWORDS = {

    "ai": ["technology", "momentum", "growth", "capex"],

    "cloud": ["margin", "growth", "technology"],

    "rate": ["financial", "small", "value", "macro"],

    "small": ["iwm", "value", "recovery"],

    "margin": ["fundamental", "profit", "quality"],

    "energy": ["xle", "oil", "commodity"],

}





def generate_thesis_insights(

    thesis_text: str,

    ticker: str,

    analysis: dict | None,

) -> dict:

    text_lower = thesis_text.lower()

    supporting: list[str] = []

    contradicting: list[str] = []

    risks: list[str] = []

    driver = "Brak powiązania z danymi — uruchom analizę dla powiązanych tickerów."



    region = (analysis or {}).get("market_region", "USA")

    gpw_extra = list(GPW_RISK_HINTS) if region == "POLAND" else []



    if not analysis:

        return {

            "supporting": [f"Uruchom analizę dla {ticker}, aby porównać z tezą."],

            "contradicting": [],

            "structural_risks": gpw_extra or ["Brak analizy w historii."],

            "likely_driver": driver,

            "gpw_structural_risks": gpw_extra,

        }



    exp = analysis.get("explanation") or {}

    region = analysis.get("market_region", region)

    gpw_extra = list(exp.get("gpw_structural_risks") or GPW_RISK_HINTS) if region == "POLAND" else []

    decision = analysis.get("decision", "WATCH")

    structural = analysis.get("structural") or {}

    technical = analysis.get("technical") or {}

    fundamental = analysis.get("fundamental") or {}

    s_class = structural.get("classification", "NOISE")



    driver = exp.get("what_is_driving_this") or exp.get("likely_driver") or driver

    risks = list(exp.get("key_risks") or [])



    if decision in ("BUY", "WATCH"):

        supporting.append(

            f"Ostatnia spójność: **{decision_label(decision)}** "

            f"(wynik {analysis.get('final_score', 0):.0%})."

        )

    if fundamental.get("score_fundamental", 0) > 0.2:

        supporting.append(

            "Fundamenty lekko wspierają: " + fundamental.get("summary", "")

        )

    if technical.get("trend") == "uptrend":

        supporting.append("Trend cenowy zgadza się z byczą narracją.")

    if s_class == "TRUE_SIGNAL":

        supporting.append("Część struktury zostaje po korekcie ruchu rynku.")



    if decision == "IGNORE":

        contradicting.append(

            f"Aktualna analiza: {decision_label('IGNORE')} — teza może wyprzedzać dowody."

        )

    if region == "POLAND":

        diag = exp.get("gpw_diagnostics") or {}

        if diag.get("speculation_risk") == "HIGH":

            contradicting.append(

                "GPW: rajd może być spekulacyjny / na wolumenie, a nie na fundamentach."

            )

        if diag.get("liquidity_risk") == "HIGH":

            contradicting.append("GPW: niska płynność — trudna realizacja tezy.")

        for w in exp.get("gpw_warnings", []):

            contradicting.append(w)

    if s_class == "MARKET_EXPOSURE":

        contradicting.append("Ruch to prawdopodobnie beta rynku/sektora, nie Twoja historia.")

    if s_class in ("FRAGILE", "NARRATIVE_DRIVEN"):

        contradicting.append(

            f"Flaga strukturalna GPW: {classification_label(s_class)}."

        )

    if s_class == "NOISE":

        contradicting.append("Brak stabilnego związku między ceną a tym typem pomysłu.")

    if technical.get("trend") == "downtrend" and any(

        w in text_lower for w in ("rally", "growth", "breakout", "higher", "wzrost", "rajd")

    ):

        contradicting.append("Oczekujesz wzrostu, a trend jest obecnie spadkowy.")



    for kw, hints in _KEYWORDS.items():

        if kw in text_lower:

            if s_class == "MARKET_EXPOSURE":

                contradicting.append(

                    f"Teza wspomina „{kw}”, ale struktura wygląda na szeroką ekspozycję."

                )

            elif fundamental.get("score_fundamental", 0) > 0:

                supporting.append(f"Temat „{kw}” nie jest sprzeczny z odczytem fundamentalnym.")



    if not supporting:

        supporting.append("Brak mocnego potwierdzenia — traktuj tezę jako niezweryfikowaną.")

    if not contradicting:

        contradicting.append("Brak dużej sprzeczności — i tak zweryfikuj na świeżych danych.")



    all_risks = list(dict.fromkeys(risks[:5] + gpw_extra))[:8]



    return {

        "supporting": supporting,

        "contradicting": contradicting,

        "structural_risks": all_risks,

        "likely_driver": driver,

        "linked_decision": decision,

        "linked_structural_class": s_class,

        "gpw_structural_risks": gpw_extra,

    }


