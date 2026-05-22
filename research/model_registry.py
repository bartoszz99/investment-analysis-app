"""Git-style experiment / model registry with content hashes."""

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def _hash_obj(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


@dataclass
class ModelRecord:
    experiment_id: str
    timestamp: str
    params_hash: str
    feature_set_hash: str
    features: list[str]
    model_params: dict
    train_period: str
    test_period: str
    oos_metrics: dict
    sharpe: float
    turnover: float
    drawdown: float
    leakage_score: float
    parent_id: str | None = None
    notes: str = ""


class ModelRegistry:
    def __init__(self, jsonl_path: str = "research/models.jsonl", parquet_path: str = "research/models_meta.parquet") -> None:
        self.jsonl_path = Path(jsonl_path)
        self.parquet_path = Path(parquet_path)
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    def register(
        self,
        features: list[str],
        model_params: dict,
        train_period: tuple[str, str],
        test_period: tuple[str, str],
        oos_metrics: dict,
        sharpe: float = 0.0,
        turnover: float = 0.0,
        drawdown: float = 0.0,
        leakage_score: float = 0.0,
        parent_id: str | None = None,
        notes: str = "",
    ) -> ModelRecord:
        record = ModelRecord(
            experiment_id=str(uuid.uuid4())[:12],
            timestamp=datetime.now(timezone.utc).isoformat(),
            params_hash=_hash_obj(model_params),
            feature_set_hash=_hash_obj(sorted(features)),
            features=features,
            model_params=model_params,
            train_period=f"{train_period[0]}..{train_period[1]}",
            test_period=f"{test_period[0]}..{test_period[1]}",
            oos_metrics=oos_metrics,
            sharpe=sharpe,
            turnover=turnover,
            drawdown=drawdown,
            leakage_score=leakage_score,
            parent_id=parent_id,
            notes=notes,
        )
        with open(self.jsonl_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(record)) + "\n")
        self._sync_parquet()
        return record

    def _sync_parquet(self) -> None:
        rows = []
        if self.jsonl_path.exists():
            for line in self.jsonl_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rows.append(json.loads(line))
        if not rows:
            return
        df = pd.DataFrame(rows)
        try:
            df.to_parquet(self.parquet_path, index=False)
        except ImportError:
            df.to_csv(self.parquet_path.with_suffix(".csv"), index=False)

    def load_all(self) -> list[ModelRecord]:
        if not self.jsonl_path.exists():
            return []
        return [ModelRecord(**json.loads(line)) for line in self.jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
