"""
Memo review workflow — track whether thesis is playing out.
"""

from __future__ import annotations

from datetime import datetime, timezone

from investment_app.memo.store import _load_raw, _save, get_memo

REVIEW_OPTIONS = {
    "thesis_playing_out": "Thesis playing out",
    "unchanged": "Unchanged",
    "weakening": "Weakening",
    "broken": "Broken",
}


def submit_review(
    memo_id: str,
    *,
    review_status: str,
    review_notes: str,
    lessons_learned: str,
) -> None:
    if review_status not in REVIEW_OPTIONS:
        raise ValueError(f"Invalid review status: {review_status}")

    memo = get_memo(memo_id)
    if not memo:
        raise ValueError("Memo not found")

    data = _load_raw()
    for m in data["memos"]:
        if m["id"] == memo_id:
            m["review_status"] = review_status
            m["review_notes"] = review_notes.strip()
            m["lessons_learned"] = lessons_learned.strip()
            m["reviewed_at"] = datetime.now(timezone.utc).isoformat()
            break
    _save(data)
