"""
Polskie tłumaczenia UI i tekstów wyświetlanych użytkownikowi.

Logika (BUY/WATCH/IGNORE, klasyfikacje, klucze danych) pozostaje po angielsku.
"""

from __future__ import annotations

TEXTS: dict[str, str] = {
    # --- Aplikacja ---
    "app.title": "Asystent analizy inwestycyjnej",
    "app.subtitle": (
        "**Osobista przestrzeń inwestycyjna** do dyscypliny w rozumowaniu — "
        "nie bot tradingowy ani silnik alfy."
    ),
    "app.table_intro": "| Strona | Cel |",
    "app.page.memo": "**Notatki inwestycyjne** | Centrum aplikacji — teza i przegląd |",
    "app.page.analyze": "**Analiza** | USA lub **Polska (GPW)** — spójność + płynność/speculacja |",
    "app.page.watchlist": "**Lista obserwowanych** | Monitorowanie rozważanych spółek |",
    "app.page.journal": "**Dziennik** | Historia rozumowania w czasie |",
    "app.page.thesis": "**Tezy** | Krótkie przekonania powiązane z dowodami |",
    "app.success.start_memo": (
        "Zacznij od **Notatek inwestycyjnych**, zanim uznać pomysł za poważne przekonanie."
    ),
    "app.info.analyze": "Użyj **Analizy** na pasku bocznym, aby szybko sprawdzić spójność.",
    # --- Decyzje (wyświetlanie) ---
    "decision.BUY": "KUP",
    "decision.WATCH": "OBSERWUJ",
    "decision.IGNORE": "ODRZUĆ",
    "decision.all": "Wszystkie",
    # --- Klasyfikacje strukturalne ---
    "classification.TRUE_SIGNAL": "sygnał strukturalny",
    "classification.STRUCTURAL_HYPOTHESIS": "hipoteza strukturalna",
    "classification.MARKET_EXPOSURE": "ekspozycja rynkowa",
    "classification.NOISE": "szum",
    "classification.ACCEPTABLE": "akceptowalna",
    "classification.FRAGILE": "krucha",
    "classification.NARRATIVE_DRIVEN": "narracja",
    "classification.LIQUIDITY_RISK": "ryzyko płynności",
    # --- Region, typy ---
    "region.USA": "Stany Zjednoczone",
    "region.POLAND": "Polska (GPW)",
    "region.all": "Wszystkie",
    "asset.stock": "Akcja",
    "asset.etf": "ETF",
    "idea.momentum": "momentum",
    "idea.value": "wartość",
    "idea.earnings": "wyniki",
    "idea.breakout": "wybicie",
    "idea.macro": "makro",
    "horizon.short": "krótki",
    "horizon.medium": "średni",
    "horizon.long": "długi",
    "gauge.strong": "Silny",
    "gauge.moderate": "Umiarkowany",
    "gauge.weak": "Słaby",
    "risk.HIGH": "WYSOKIE",
    "risk.MEDIUM": "ŚREDNIE",
    "risk.LOW": "NISKIE",
    "risk.MODERATE": "UMIARKOWANE",
    "trend.uptrend": "wzrost",
    "trend.downtrend": "spadek",
    "trend.mixed": "mieszany",
    "trend.unknown": "nieznany",
    "vol.high": "wysoka",
    "vol.low": "niska",
    "vol.normal": "normalna",
    "vol.unknown": "nieznana",
    # --- ui_common / formularz ---
    "ui.market_region": "Rynek",
    "ui.your_idea": "Twój pomysł",
    "ui.gpw_examples": "Przykłady GPW",
    "ui.ticker": "Ticker",
    "ui.asset_type": "Typ aktywa",
    "ui.idea_type": "Typ pomysłu",
    "ui.time_horizon": "Horyzont czasowy",
    "ui.user_note": "Twoja notatka (opcjonalnie)",
    "ui.user_note_ph": "Dlaczego patrzysz na ten instrument?",
    "ui.analyze_btn": "Analizuj",
    "ui.enter_ticker": "Podaj ticker.",
    "ui.analyzing": "Analizuję {ticker} ({region})…",
    "ui.coherence_score": "Wynik spójności",
    "ui.fundamental": "Fundamenty",
    "ui.technical": "Technika",
    "ui.structural": "Struktura",
    "ui.final_score": "Wynik końcowy",
    "ui.gpw_diagnostics": "Diagnostyka strukturalna GPW",
    "ui.liquidity_risk": "Ryzyko płynności",
    "ui.speculation_risk": "Ryzyko spekulacji",
    "ui.narrative_dependence": "Zależność od narracji",
    "ui.ownership_flags": "Flagi właścicielskie / koncentracja",
    "ui.driving_idea": "Co prawdopodobnie napędza ten pomysł?",
    "ui.three_axis": "Podsumowanie trzech osi",
    "ui.fundamentals_axis": "Fundamenty",
    "ui.technicals_axis": "Technika",
    "ui.structure_axis": "Struktura",
    "ui.trend_vol": "Trend: **{trend}** · Zmienność: **{vol}**",
    "ui.counterparty": "Kto jest po drugiej stronie?",
    "ui.key_risks": "Kluczowe ryzyka",
    "ui.price_context": "Kontekst ceny",
    "ui.chart_unavailable": "Wykres niedostępny.",
    # --- Analiza (strona) ---
    "page.analyze.title": "Analiza pomysłu",
    "page.analyze.caption": (
        "Czy to rozumowanie inwestycyjne jest spójne? Nie: czy pobije rynek?"
    ),
    "page.analyze.saved_journal": "Zapisano w dzienniku inwestycyjnym.",
    "page.analyze.add_watchlist": "Dodaj do listy obserwowanych",
    "page.analyze.added_watchlist": "{ticker} dodano do listy obserwowanych.",
    "page.analyze.create_memo": "Utwórz notatkę inwestycyjną",
    "page.analyze.buy_hint": (
        "KUP z modelu oznacza tylko spójność — napisz notatkę przed realnym przekonaniem."
    ),
    # --- Lista obserwowanych ---
    "page.watchlist.title": "Lista obserwowanych",
    "page.watchlist.caption": "Panel monitorowania — nie skaner tradingowy.",
    "page.watchlist.add_ticker": "Dodaj ticker",
    "page.watchlist.add_ph": "np. CDR lub MSFT",
    "page.watchlist.type": "Typ",
    "page.watchlist.idea": "Pomysł",
    "page.watchlist.horizon": "Horyzont",
    "page.watchlist.add_btn": "Dodaj do listy",
    "page.watchlist.added": "Dodano {ticker}",
    "page.watchlist.refresh_all": "Odśwież wszystko",
    "page.watchlist.refreshing": "Ponowna analiza listy…",
    "page.watchlist.empty": (
        "Lista jest pusta. Dodaj tickery z Analizy lub powyżej."
    ),
    "page.watchlist.click_refresh": "Kliknij **Odśwież wszystko**, aby wczytać najnowsze wyniki.",
    "page.watchlist.not_refreshed": "nie odświeżono",
    "page.watchlist.score_change": "Zmiana wyniku: {delta}",
    "page.watchlist.trend": "Trend: {trend}",
    "page.watchlist.remove": "Usuń",
    "page.watchlist.refresh_failed": "Odświeżanie nie powiodło się: {error}",
    # --- Dziennik ---
    "page.journal.title": "Dziennik inwestycyjny",
    "page.journal.caption": "Śledzenie rozumowania — nie PnL portfela.",
    "page.journal.ticker_filter": "Ticker",
    "page.journal.decision_filter": "Decyzja",
    "page.journal.region_filter": "Rynek",
    "page.journal.filter_from_date": "Filtruj od daty",
    "page.journal.from_date": "Od",
    "page.journal.empty": (
        "Brak wpisów. Uruchom analizę na stronie **Analiza**."
    ),
    "page.journal.entries_shown": "Wyświetlone wpisy",
    "page.journal.structure": "Struktura: {cls}",
    "page.journal.price_at": "Cena przy analizie",
    "page.journal.price_now": "Cena bieżąca",
    "page.journal.change_since": "Zmiana od analizy",
    "page.journal.followup": "Kontrola:",
    "page.journal.your_note": "*Twoja notatka:*",
    # --- Tezy ---
    "page.thesis.title": "Tezy",
    "page.thesis.caption": "Dlaczego w to wierzysz? Co by to obaliło?",
    "page.thesis.new_expander": "Nowa teza",
    "page.thesis.short_title": "Krótki tytuł (opcjonalnie)",
    "page.thesis.my_thesis": "Moja teza",
    "page.thesis.my_thesis_ph": (
        "np. cykl nakładów na AI nadal wspiera łańcuch dostaw półprzewodników"
    ),
    "page.thesis.linked_tickers": "Powiązane tickery (po przecinku)",
    "page.thesis.linked_ph": "NVDA, AMD, AVGO",
    "page.thesis.save": "Zapisz tezę",
    "page.thesis.saved": "Teza zapisana.",
    "page.thesis.empty": "Brak tez. Napisz jedną powyżej.",
    "page.thesis.select": "Wybierz tezę",
    "page.thesis.tickers": "Tickery: {tickers}",
    "page.thesis.tickers_none": "brak",
    "page.thesis.delete": "Usuń tę tezę",
    "page.thesis.evidence": "Weryfikacja dowodów",
    "page.thesis.run_live": "Uruchom świeżą analizę dla {ticker}",
    "page.thesis.analyzing": "Analizuję…",
    "page.thesis.supporting": "Potwierdzające",
    "page.thesis.contradicting": "Sprzeczne",
    "page.thesis.structural_risks": "Ryzyka strukturalne",
    "page.thesis.gpw_risks": "Ryzyka strukturalne specyficzne dla GPW",
    "page.thesis.likely_driver": "**Prawdopodobny motor:**",
    # --- Notatki ---
    "page.memo.title": "Notatki inwestycyjne",
    "page.memo.intro": (
        "Opisz **dlaczego** wierzysz w pomysł — i **co by go obaliło**. "
        "To dyscyplina myślenia, nie prognoza ceny."
    ),
    "page.memo.reminder": (
        "Zanim uznać pomysł za poważne **KUP**, napisz notatkę, którą ktoś inny "
        "mógłby zakwestionować. Jeśli nie potrafisz wskazać, co **obali** tezę, "
        "pomysł nie jest gotowy na przekonanie."
    ),
    "page.memo.tab.create": "Nowa notatka",
    "page.memo.tab.active": "Aktywne notatki",
    "page.memo.tab.review": "Przegląd",
    "page.memo.guide_title": "Poradnik — słabe vs mocne",
    "page.memo.weak": "Słabe: \"{text}\"",
    "page.memo.strong": "Mocne: *{text}*",
    "page.memo.ticker_req": "Ticker *",
    "page.memo.title_req": "Tytuł tezy *",
    "page.memo.summary_req": "Podsumowanie tezy *",
    "page.memo.summary_ph": "Co musi być prawdą, żeby ta inwestycja miała sens?",
    "page.memo.driver_req": "Oczekiwany motor *",
    "page.memo.driver_ph": "Kto jest zmuszony do działania i dlaczego?",
    "page.memo.mispricing": "Czego rynek może nie widzieć?",
    "page.memo.risks_req": "Kluczowe ryzyka *",
    "page.memo.invalidation_req": "Co obaliłoby tezę? *",
    "page.memo.invalidation_ph": "Bądź konkretny — daty, wskaźniki, wydarzenia.",
    "page.memo.why_now": "Dlaczego teraz?",
    "page.memo.valuation": "Logika wyceny",
    "page.memo.conviction": "Przekonanie (Twoja ocena, nie model)",
    "page.memo.linked_analysis": (
        "Powiązana analiza: **{decision}** ({score}) — {summary}"
    ),
    "page.memo.clarity_draft": "**Jasność (szkic):** {label} ({score})",
    "page.memo.metric.specificity": "Konkretność",
    "page.memo.metric.falsifiable": "Możliwość obalenia",
    "page.memo.metric.risks": "Ryzyka",
    "page.memo.metric.horizon": "Horyzont",
    "page.memo.metric.driver": "Motor",
    "page.memo.save": "Zapisz notatkę",
    "page.memo.save_clear": "Zapisz i wyczyść szablon",
    "page.memo.required_error": (
        "Ticker, tytuł, podsumowanie tezy i warunki obalenia są wymagane."
    ),
    "page.memo.saved": "Notatka zapisana ({label}).",
    "page.memo.no_memos": "Brak notatek. Utwórz jedną w zakładce **Nowa notatka**.",
    "page.memo.analysis_line": "Analiza: **{decision}** · Przekonanie **{confidence}**",
    "page.memo.linked_score": "Powiązany wynik: {score}",
    "page.memo.review_status": "Przegląd: {status}",
    "page.memo.open": "Otwórz",
    "page.memo.detail_title": "Notatka: {ticker}",
    "page.memo.clarity": "Jasność: **{label}** ({score}) · Utworzono {date}",
    "page.memo.summary": "Podsumowanie",
    "page.memo.expected_driver": "Oczekiwany motor",
    "page.memo.market_missing": "Czego rynek może nie widzieć",
    "page.memo.valuation_h": "Wycena",
    "page.memo.key_risks_h": "Kluczowe ryzyka",
    "page.memo.invalidation_h": "Warunki obalenia",
    "page.memo.why_now_h": "Dlaczego teraz",
    "page.memo.linked_note": "Powiązana notatka z analizy: {text}",
    "page.memo.save_review_first": "Najpierw zapisz notatkę.",
    "page.memo.pick_review": "Notatka do przeglądu",
    "page.memo.thesis_snip": "**Teza:** {text}…",
    "page.memo.how_thesis": "Jak radzi sobie teza?",
    "page.memo.what_right": "Co było trafne?",
    "page.memo.what_right_ph": "Jakie dowody Cię wspierały?",
    "page.memo.what_wrong": "Co było błędne?",
    "page.memo.what_wrong_ph": "Co źle odczytałeś?",
    "page.memo.market_cared": "Na czym skupił się rynek?",
    "page.memo.market_cared_ph": "np. stopy, rotacja sektorowa, zły wynik",
    "page.memo.lessons": "Wnioski",
    "page.memo.gpw_review": "Przypomnienia przeglądu GPW",
    "page.memo.gpw_review_bullets": (
        "- Obserwuj pogorszenie płynności lub rozszerzenie spreadów\n"
        "- Wzrost spekulacji (skok wolumenu bez fundamentów)\n"
        "- Rajd na narracji bez wsparcia w wynikach"
    ),
    "page.memo.save_review": "Zapisz przegląd",
    "page.memo.review_saved": "Przegląd zapisany.",
    "page.memo.prev_review": "Poprzednie notatki z przeglądu",
    # --- Przegląd notatek ---
    "review.thesis_playing_out": "Teza się realizuje",
    "review.unchanged": "Bez zmian",
    "review.weakening": "Osłabia się",
    "review.broken": "Obalona",
    # --- Jasność notatki ---
    "clarity.WEAK THESIS": "SŁABA TEZA",
    "clarity.ACCEPTABLE": "AKCEPTOWALNA",
    "clarity.STRONGLY ARTICULATED": "DOBRZE UZASADNIONA",
    # --- Flagi właścicielskie (ticker_mapper → PL) ---
    "flag.state_linked_bank": "bank powiązany z państwem",
    "flag.policy_sensitive": "wrażliwość na politykę",
    "flag.state_energy": "energetyka pod kontrolą państwa",
    "flag.regulated_pricing": "ryzyko regulacji cen",
    "flag.state_insurer": "ubezpieczyciel powiązany z państwem",
    "flag.state_miner": "kopalnia powiązana z państwem",
    "flag.state_refiner": "rafineria powiązana z państwem",
    "flag.cyclical_rates": "cykliczność / stopy / surowce",
}

# Mapowanie angielskich flag z ticker_mapper
FLAG_EN_TO_KEY: dict[str, str] = {
    "state-linked bank": "flag.state_linked_bank",
    "policy-sensitive": "flag.policy_sensitive",
    "state-controlled energy": "flag.state_energy",
    "regulated pricing risk": "flag.regulated_pricing",
    "state-linked insurer": "flag.state_insurer",
    "state-linked miner": "flag.state_miner",
    "state-linked refiner": "flag.state_refiner",
    "cyclical / commodity or rates sensitivity": "flag.cyclical_rates",
}


def translate(key: str, **kwargs) -> str:
    """Zwraca polski tekst dla klucza; brak klucza → zwraca key."""
    text = TEXTS.get(key, key)
    if kwargs:
        try:
            return text.format(**kwargs)
        except KeyError:
            return text
    return text


def decision_label(decision: str | None) -> str:
    if not decision or decision == "All":
        return translate("decision.all") if decision == "All" else (decision or "—")
    return translate(f"decision.{decision}")


def classification_label(cls: str | None) -> str:
    if not cls:
        return "—"
    return translate(f"classification.{cls}")


def region_label(region: str) -> str:
    return translate(f"region.{region}")


def asset_label(asset: str) -> str:
    m = {"Stock": "asset.stock", "ETF": "asset.etf"}
    return translate(m.get(asset, asset))


def idea_label(idea: str) -> str:
    return translate(f"idea.{idea.lower()}")


def horizon_label(h: str) -> str:
    return translate(f"horizon.{h.lower()}")


def risk_level_label(level: str | None) -> str:
    if not level:
        return "—"
    return translate(f"risk.{level}")


def trend_label(trend: str | None) -> str:
    if not trend:
        return "—"
    return translate(f"trend.{trend.lower()}")


def vol_label(vol: str | None) -> str:
    if not vol:
        return "—"
    return translate(f"vol.{vol.lower()}")


def gauge_label(score: float) -> str:
    if score >= 0.65:
        return translate("gauge.strong")
    if score >= 0.4:
        return translate("gauge.moderate")
    return translate("gauge.weak")


def clarity_label(label: str) -> str:
    return translate(f"clarity.{label}")


def review_label(status_key: str) -> str:
    return translate(f"review.{status_key}")


def translate_flag(en_text: str) -> str:
    key = FLAG_EN_TO_KEY.get(en_text.lower().strip())
    if key:
        return translate(key)
    return en_text
