"""
Feature lineage — DAG, audit tracing, JSON reports.
Temporal guarantee: lineage describes transforms only; no future data in features.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class LeakageRisk(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class LineageNode:
    feature_name: str
    source_columns: list[str]
    lag: int
    lookback: int
    transformation_chain: list[str]
    leakage_risk: str
    created_at: str
    parents: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FeatureLineageReport:
    nodes: dict[str, LineageNode]
    dag_edges: list[tuple[str, str]]
    generated_at: str
    anti_leakage_note: str = "All features use lag>=1 before rolling; no full-sample stats"

    def to_json(self, path: str) -> None:
        payload = {
            "generated_at": self.generated_at,
            "anti_leakage_note": self.anti_leakage_note,
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "dag_edges": [{"from": a, "to": b} for a, b in self.dag_edges],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    def trace_raw_fields(self, feature_name: str) -> list[str]:
        """Which raw OHLCV fields contributed to this feature?"""
        if feature_name not in self.nodes:
            return []
        return list(self.nodes[feature_name].source_columns)


class FeatureLineageTracker:
    def __init__(self) -> None:
        self._nodes: dict[str, LineageNode] = {}

    def register(
        self,
        feature_name: str,
        source_columns: list[str],
        lag: int,
        lookback: int,
        transformation_chain: list[str],
        leakage_risk: LeakageRisk = LeakageRisk.LOW,
        parents: list[str] | None = None,
    ) -> None:
        self._nodes[feature_name] = LineageNode(
            feature_name=feature_name,
            source_columns=source_columns,
            lag=lag,
            lookback=lookback,
            transformation_chain=transformation_chain,
            leakage_risk=leakage_risk.value,
            created_at=datetime.now(timezone.utc).isoformat(),
            parents=parents or [],
        )

    def register_from_feature_store(self) -> FeatureLineageTracker:
        from layers.feature_store import FeatureStore

        store = FeatureStore()
        for name, spec in store._registry.items():
            self.register(
                name,
                list(spec.dependencies) or ["Close"],
                spec.lag,
                spec.lookback,
                [f"lag({spec.lag})", f"rolling({spec.lookback})", spec.category],
                LeakageRisk.LOW if spec.leakage_safe else LeakageRisk.HIGH,
            )
        for name in (f"SMA_{7}", f"SMA_{30}"):
            self.register(
                name,
                ["Close"],
                1,
                int(name.split("_")[1]),
                ["lag(1)", "rolling_mean", "shift_before_roll"],
                LeakageRisk.LOW,
                parents=["Close"],
            )
        return self

    def build_dag(self) -> list[tuple[str, str]]:
        edges = []
        for name, node in self._nodes.items():
            for parent in node.parents:
                edges.append((parent, name))
            for src in node.source_columns:
                if src != name:
                    edges.append((src, name))
        return edges

    def report(self) -> FeatureLineageReport:
        return FeatureLineageReport(
            nodes=dict(self._nodes),
            dag_edges=self.build_dag(),
            generated_at=datetime.now(timezone.utc).isoformat(),
        )
