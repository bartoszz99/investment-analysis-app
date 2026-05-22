"""
Wskazówki UI — słabe vs mocne pisanie notatek (bez AI).
"""

from __future__ import annotations

GUIDANCE_BLOCKS: list[dict] = [
    {
        "field": "Podsumowanie tezy",
        "bad": "akcja AI idzie w górę",
        "good": (
            "Przyspieszenie przychodów z capex AI w enterprise może szybciej "
            "przewartościować marże niż konsensus."
        ),
    },
    {
        "field": "Warunki obalenia",
        "bad": "wygląda byczo",
        "good": "Jeśli marże pogorszą się przez 2 kolejne kwartały, teza pada.",
    },
    {
        "field": "Oczekiwany motor",
        "bad": "rynek poleci",
        "good": (
            "Realokacje instytucji w jakościowy wzrost po rewizjach wyników — "
            "kupujący z przymusu, nie hype."
        ),
    },
    {
        "field": "Czego rynek może nie widzieć",
        "bad": "niedowartościowana",
        "good": (
            "Modele zakładają płaskie capex; kanały sugerują 15–20% wyższe "
            "wydatki do końca roku."
        ),
    },
    {
        "field": "Kluczowe ryzyka",
        "bad": "zmienność",
        "good": (
            "Regulacja cen; koncentracja klientów; kompresja wielokrotności "
            "przy wyższych stopach."
        ),
    },
]


BUY_MEMO_REMINDER = (
    "Zanim uznać pomysł za poważne **KUP**, napisz notatkę, którą ktoś inny "
    "mógłby zakwestionować. Jeśli nie potrafisz wskazać, co **obali** tezę, "
    "pomysł nie jest gotowy na przekonanie."
)
