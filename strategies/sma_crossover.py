import pandas as pd

import config
from layers.feature_engine import FeatureEngine
from strategies.base import BaseStrategy


class SmaCrossover(BaseStrategy):
    def __init__(self, sma_short: int | None = None, sma_long: int | None = None):
        self.sma_short = sma_short if sma_short is not None else config.SMA_SHORT
        self.sma_long = sma_long if sma_long is not None else config.SMA_LONG
        self._engine = FeatureEngine()

    @property
    def name(self) -> str:
        return f"SMA {self.sma_short}/{self.sma_long}"

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        result = self._engine.add_sma_features(df, self.sma_short, self.sma_long)
        short_col = f"SMA_{self.sma_short}"
        long_col = f"SMA_{self.sma_long}"

        valid = result[short_col].notna() & result[long_col].notna()
        result["signal"] = ""
        result.loc[valid & (result[short_col] > result[long_col]), "signal"] = "BUY"
        result.loc[valid & (result[short_col] < result[long_col]), "signal"] = "SELL"

        raw_position = pd.Series(index=result.index, dtype=float)
        raw_position.loc[result["signal"] == "BUY"] = 1
        raw_position.loc[result["signal"] == "SELL"] = 0
        result["position"] = raw_position.ffill().fillna(0).astype(int)
        result["signal_weight"] = result["position"].astype(float)

        return result
