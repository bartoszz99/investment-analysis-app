"""Incremental feature cache — recomputes only when input hash changes."""

import hashlib
from typing import Any

import pandas as pd


class FeatureCache:
    def __init__(self) -> None:
        self._store: dict[str, pd.DataFrame] = {}

    @staticmethod
    def _key(df: pd.DataFrame, names: tuple[str, ...]) -> str:
        tail = df[["Close"]].tail(5).to_json() if "Close" in df.columns else ""
        raw = f"{len(df)}|{df.index[-1] if len(df) else ''}|{names}|{tail}"
        return hashlib.md5(raw.encode()).hexdigest()

    def get_or_build(
        self,
        df: pd.DataFrame,
        names: list[str],
        builder,
    ) -> pd.DataFrame:
        key = self._key(df, tuple(sorted(names)))
        if key in self._store:
            return self._store[key]
        built = builder(df, names)
        self._store[key] = built
        return built

    def invalidate(self) -> None:
        self._store.clear()
