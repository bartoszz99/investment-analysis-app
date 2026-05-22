"""
Memo persistence — investment_app/memo/memos.json
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from investment_app.data.market_region import parse_region
from investment_app.data.ticker_mapper import display_ticker, normalize_ticker
from investment_app.engine import APP_DIR
from investment_app.memo.schema import InvestmentMemo
from investment_app.memo.scoring import score_memo_clarity

MEMO_DIR = APP_DIR / "memo"
MEMOS_FILE = MEMO_DIR / "memos.json"


def _ensure() -> None:
    MEMO_DIR.mkdir(parents=True, exist_ok=True)
    if not MEMOS_FILE.exists():
        _save({"memos": []})
        seed_examples()


def _load_raw() -> dict:
    _ensure()
    with open(MEMOS_FILE, encoding="utf-8") as f:
        return json.load(f)


def _save(data: dict) -> None:
    MEMO_DIR.mkdir(parents=True, exist_ok=True)
    with open(MEMOS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def list_memos(*, ticker: str | None = None) -> list[InvestmentMemo]:
    data = _load_raw()
    memos = [InvestmentMemo.from_dict(m) for m in data.get("memos", [])]
    if ticker:
        memos = [m for m in memos if m.ticker.upper() == ticker.upper()]
    memos.sort(key=lambda m: m.created_at, reverse=True)
    return memos


def get_memo(memo_id: str) -> InvestmentMemo | None:
    for m in list_memos():
        if m.id == memo_id:
            return m
    return None


def save_memo(fields: dict) -> InvestmentMemo:
    clarity = score_memo_clarity(fields)
    region = parse_region(fields.get("market_region"))
    sym = normalize_ticker(fields.get("ticker", ""), region)
    disp = display_ticker(sym)
    record = InvestmentMemo(
        id=str(uuid.uuid4())[:8],
        created_at=datetime.now(timezone.utc).isoformat(),
        ticker=sym,
        market_region=region,
        display_ticker=disp,
        thesis_title=fields.get("thesis_title", "").strip(),
        thesis_summary=fields.get("thesis_summary", "").strip(),
        expected_driver=fields.get("expected_driver", "").strip(),
        market_mispricing=fields.get("market_mispricing", "").strip(),
        key_risks=fields.get("key_risks", "").strip(),
        invalidation_conditions=fields.get("invalidation_conditions", "").strip(),
        time_horizon=fields.get("time_horizon", "medium").strip(),
        valuation_case=fields.get("valuation_case", "").strip(),
        why_now=fields.get("why_now", "").strip(),
        confidence_0_100=int(fields.get("confidence_0_100", 50)),
        linked_analysis_score=fields.get("linked_analysis_score"),
        linked_decision=fields.get("linked_decision"),
        linked_explanation_summary=fields.get("linked_explanation_summary"),
        clarity_label=clarity["clarity_label"],
        clarity_score=clarity["clarity_score"],
        clarity_breakdown=clarity["clarity_breakdown"],
    )
    data = _load_raw()
    data["memos"].append(record.to_dict())
    _save(data)
    return record


def ensure_examples_if_empty() -> None:
    data = _load_raw()
    if not data.get("memos"):
        seed_examples()


def seed_examples() -> None:
    """Example memos for onboarding — not predictions."""
    examples = [
        {
            "ticker": "NVDA",
            "thesis_title": "Enterprise AI capex cycle",
            "thesis_summary": (
                "Hyperscaler and enterprise capex on AI infrastructure remains above "
                "consensus through next 4 quarters, supporting datacenter revenue mix "
                "and margin durability versus CPU-only models."
            ),
            "expected_driver": (
                "Institutional reallocations into AI enablers after earnings revisions; "
                "supply constraints keep pricing power elevated."
            ),
            "market_mispricing": (
                "Street models assume capex flattening in H2; channel data suggests "
                "continued acceleration in accelerator orders."
            ),
            "key_risks": (
                "Export controls; customer concentration; multiple compression if "
                "rates stay higher; competition from in-house silicon."
            ),
            "invalidation_conditions": (
                "Two consecutive quarters of datacenter revenue growth below 10% YoY "
                "OR guide-down on capex from two top customers."
            ),
            "time_horizon": "medium",
            "valuation_case": (
                "Premium multiple justified only if growth sustains; compare EV/sales "
                "to historical band when growth decelerates."
            ),
            "why_now": (
                "Post-earnings drift window; revisions trend still upward; "
                "technical trend intact vs SPY."
            ),
            "confidence_0_100": 62,
            "linked_decision": "WATCH",
            "linked_analysis_score": 0.48,
        },
        {
            "ticker": "IWM",
            "thesis_title": "Small-cap rate relief",
            "thesis_summary": (
                "If policy rates peak and financial conditions ease, small caps "
                "with domestic revenue may outperform large-cap tech concentration."
            ),
            "expected_driver": (
                "Passive and factor rebalancers rotating from mega-cap growth "
                "into broader participation indices."
            ),
            "market_mispricing": (
                "Market prices prolonged recession; credit spreads already reflect "
                "severe stress relative to labor data."
            ),
            "key_risks": (
                "Recession deeper than expected; regional bank stress; "
                "liquidity gaps in small caps."
            ),
            "invalidation_conditions": (
                "IWM underperforms SPY by >8% over 3 months while real rates rise "
                "OR credit spreads widen sharply."
            ),
            "time_horizon": "long",
            "valuation_case": (
                "Relative valuation vs large caps at historical wide spread; "
                "requires catalyst not just cheapness."
            ),
            "why_now": (
                "Breadth improving in recent rally attempts; monitoring "
                "rate-sensitive leadership."
            ),
            "confidence_0_100": 45,
            "linked_decision": "WATCH",
            "linked_analysis_score": 0.39,
        },
    ]
    data = {"memos": []}
    for ex in examples:
        clarity = score_memo_clarity(ex)
        data["memos"].append(
            InvestmentMemo(
                id=str(uuid.uuid4())[:8],
                created_at=datetime.now(timezone.utc).isoformat(),
                review_status="open",
                clarity_label=clarity["clarity_label"],
                clarity_score=clarity["clarity_score"],
                clarity_breakdown=clarity["clarity_breakdown"],
                **{k: v for k, v in ex.items()},
            ).to_dict()
        )
    _save(data)


def prefill_from_analysis(result) -> dict:
    """Build form defaults from AnalysisResult."""
    exp = result.explanation or {}
    return {
        "ticker": getattr(result, "display_ticker", result.ticker),
        "market_region": getattr(result, "market_region", "USA"),
        "display_ticker": getattr(result, "display_ticker", result.ticker),
        "thesis_title": f"{result.ticker} — {result.idea_type} idea",
        "thesis_summary": exp.get("what_is_driving_this", ""),
        "expected_driver": exp.get("likely_driver", ""),
        "market_mispricing": "",
        "key_risks": "\n".join(exp.get("key_risks") or []),
        "invalidation_conditions": "",
        "time_horizon": result.horizon,
        "valuation_case": "",
        "why_now": exp.get("headline", ""),
        "confidence_0_100": int(min(90, max(20, result.final_score * 100))),
        "linked_analysis_score": result.final_score,
        "linked_decision": result.decision,
        "linked_explanation_summary": exp.get("headline", ""),
    }
