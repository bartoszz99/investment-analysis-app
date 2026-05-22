"""
Journal persistence — JSONL append-only log of every analysis.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from investment_app.engine import APP_DIR, AnalysisResult

JOURNAL_DIR = APP_DIR / "journal"
ENTRIES_FILE = JOURNAL_DIR / "entries.jsonl"


def _ensure_dir() -> None:
    JOURNAL_DIR.mkdir(parents=True, exist_ok=True)


def save_analysis_to_journal(
    result: AnalysisResult,
    *,
    price_at_analysis: float | None,
    user_note: str | None = None,
) -> str:
    _ensure_dir()
    entry_id = str(uuid.uuid4())[:8]
    record = {
        "id": entry_id,
        "timestamp": result.as_of or datetime.now(timezone.utc).isoformat(),
        "ticker": result.ticker,
        "display_ticker": getattr(result, "display_ticker", result.ticker),
        "market_region": getattr(result, "market_region", "USA"),
        "asset_type": result.asset_type,
        "idea_type": result.idea_type,
        "horizon": result.horizon,
        "decision": result.decision,
        "final_score": result.final_score,
        "structural_class": result.structural.get("classification"),
        "breakdown": result.breakdown,
        "headline": result.explanation.get("headline"),
        "what_is_driving": result.explanation.get("what_is_driving_this"),
        "key_risks": result.explanation.get("key_risks", []),
        "price_at_analysis": price_at_analysis,
        "user_note": user_note or "",
        "explanation": result.explanation,
    }
    with open(ENTRIES_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
    return entry_id


def load_entries(
    *,
    ticker: str | None = None,
    decision: str | None = None,
    date_from: str | None = None,
) -> list[dict]:
    if not ENTRIES_FILE.exists():
        return []
    rows: list[dict] = []
    with open(ENTRIES_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if ticker:
        rows = [r for r in rows if r.get("ticker", "").upper() == ticker.upper()]
    if decision and decision != "All":
        rows = [r for r in rows if r.get("decision") == decision]
    if date_from:
        rows = [r for r in rows if (r.get("timestamp") or "")[:10] >= date_from]

    rows.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    return rows
