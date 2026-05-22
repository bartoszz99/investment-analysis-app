"""Reproducible experiment tracking."""

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class ExperimentRecord:
    experiment_id: str
    timestamp: str
    features: list[str]
    model_params: dict
    train_window: int
    test_window: int
    metrics: dict
    sharpe: float
    turnover: float
    drawdown: float
    leakage_score: float
    notes: str = ""


class ExperimentRegistry:
    def __init__(self, path: str = "research/experiments.jsonl") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def register(
        self,
        features: list[str],
        model_params: dict,
        train_window: int,
        test_window: int,
        metrics: dict,
        sharpe: float = 0.0,
        turnover: float = 0.0,
        drawdown: float = 0.0,
        leakage_score: float = 0.0,
        notes: str = "",
    ) -> ExperimentRecord:
        record = ExperimentRecord(
            experiment_id=str(uuid.uuid4())[:8],
            timestamp=datetime.now(timezone.utc).isoformat(),
            features=features,
            model_params=model_params,
            train_window=train_window,
            test_window=test_window,
            metrics=metrics,
            sharpe=sharpe,
            turnover=turnover,
            drawdown=drawdown,
            leakage_score=leakage_score,
            notes=notes,
        )
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(record)) + "\n")
        return record

    def load_all(self) -> list[ExperimentRecord]:
        if not self.path.exists():
            return []
        records = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(ExperimentRecord(**json.loads(line)))
        return records
