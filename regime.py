"""Regime layer facade."""
import pandas as pd

from layers.regime_layer import (
    ALL_REGIMES,
    REGIME_HIGH_VOLATILITY,
    REGIME_MEAN_REVERSION,
    REGIME_TREND,
    detect_regime as _detect,
)


def detect_regime(df: pd.DataFrame) -> pd.DataFrame:
    return _detect(df)


def regime_distribution(df: pd.DataFrame) -> pd.Series:
    if "regime" not in df.columns:
        df = detect_regime(df)
    return df["regime"].dropna().value_counts(normalize=True).sort_index() * 100


def print_regime_report(df: pd.DataFrame, label: str = "") -> None:
    if "regime" not in df.columns:
        df = detect_regime(df)
    print(f"\n=== Raport rezimow{': ' + label if label else ''} ===")
    valid = df["regime"].dropna()
    if valid.empty:
        print("Brak danych.")
        return
    dist = regime_distribution(df)
    for regime in ALL_REGIMES:
        print(f"  {regime:<18} {dist.get(regime, 0.0):6.2f}%")
