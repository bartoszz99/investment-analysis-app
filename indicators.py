"""Feature layer facade."""
import pandas as pd

from layers.feature_engine import FeatureEngine, FeatureSpec


def add_sma(df: pd.DataFrame, window: int, column: str = "Close") -> pd.DataFrame:
    engine = FeatureEngine()
    out = df.copy()
    out[f"SMA_{window}"] = engine.sma(out[column], window)
    return out


def add_moving_averages(df: pd.DataFrame, short: int, long: int) -> pd.DataFrame:
    return FeatureEngine().add_sma_features(df, short, long)
