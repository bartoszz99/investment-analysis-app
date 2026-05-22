"""
Thesis persistence — user-written investment narratives.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from investment_app.engine import APP_DIR

THESIS_DIR = APP_DIR / "thesis"
THESIS_FILE = THESIS_DIR / "theses.json"


def _load() -> dict:
    THESIS_DIR.mkdir(parents=True, exist_ok=True)
    if not THESIS_FILE.exists():
        return {"theses": []}
    with open(THESIS_FILE, encoding="utf-8") as f:
        return json.load(f)


def _save(data: dict) -> None:
    THESIS_DIR.mkdir(parents=True, exist_ok=True)
    with open(THESIS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def list_theses() -> list[dict]:
    return list(_load().get("theses", []))


def add_thesis(text: str, tickers: list[str], title: str | None = None) -> dict:
    data = _load()
    record = {
        "id": str(uuid.uuid4())[:8],
        "title": title or (text[:60] + "…" if len(text) > 60 else text),
        "text": text.strip(),
        "tickers": [t.upper().strip() for t in tickers if t.strip()],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    data["theses"].append(record)
    _save(data)
    return record


def delete_thesis(thesis_id: str) -> None:
    data = _load()
    data["theses"] = [t for t in data["theses"] if t["id"] != thesis_id]
    _save(data)


def update_thesis(thesis_id: str, *, text: str | None = None, tickers: list[str] | None = None) -> None:
    data = _load()
    for t in data["theses"]:
        if t["id"] == thesis_id:
            if text is not None:
                t["text"] = text.strip()
            if tickers is not None:
                t["tickers"] = [x.upper() for x in tickers]
            t["updated_at"] = datetime.now(timezone.utc).isoformat()
            break
    _save(data)
