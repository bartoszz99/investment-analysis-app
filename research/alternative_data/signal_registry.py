"""
Signal registry — catalog of alternative research signals.
Research-only; not wired to production portfolio.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

from research.alternative_data.base import AlternativeSignal, LeakageRisk, SignalMetadata


class SignalRegistry:
    def __init__(self) -> None:
        self._signals: dict[str, AlternativeSignal] = {}

    def register(self, signal: AlternativeSignal) -> None:
        signal.with_stats()
        self._signals[signal.metadata.name] = signal

    def get(self, name: str) -> AlternativeSignal:
        return self._signals[name]

    def list_signals(self) -> list[str]:
        return list(self._signals.keys())

    def __iter__(self) -> Iterator[AlternativeSignal]:
        return iter(self._signals.values())

    def summary_table(self) -> list[dict]:
        rows = []
        for sig in self._signals.values():
            m = sig.metadata
            rows.append(
                {
                    "name": m.name,
                    "source": m.source,
                    "lag_days": m.lag_days,
                    "update_frequency": m.update_frequency,
                    "leakage_risk": m.leakage_risk.value,
                    "timestamp_assumption": m.timestamp_assumption,
                    "lag_policy": m.lag_policy,
                    "mean": sig.stats.get("mean"),
                    "std": sig.stats.get("std"),
                    "n_obs": sig.stats.get("n_obs"),
                }
            )
        return rows

    def to_json(self, path: str | Path) -> None:
        payload = {
            "signals": [
                {
                    "metadata": {
                        **{k: v for k, v in asdict(s.metadata).items() if k != "leakage_risk"},
                        "leakage_risk": s.metadata.leakage_risk.value,
                    },
                    "stats": s.stats,
                }
                for s in self._signals.values()
            ]
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
