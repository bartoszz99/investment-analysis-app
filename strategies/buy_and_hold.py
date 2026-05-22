import pandas as pd

from strategies.base import BaseStrategy


class BuyAndHold(BaseStrategy):
    @property
    def name(self) -> str:
        return "Buy & Hold"

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        result["position"] = 1
        result["signal_weight"] = 1.0
        return result
