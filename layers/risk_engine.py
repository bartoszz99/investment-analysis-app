"""Risk monitoring and kill-switches."""

import numpy as np
import pandas as pd


class RiskEngine:
    def __init__(
        self,
        max_drawdown: float = 0.15,
        max_vol_spike: float = 2.5,
        max_concentration: float = 0.5,
        max_turnover_daily: float = 0.5,
    ) -> None:
        self.max_drawdown = max_drawdown
        self.max_vol_spike = max_vol_spike
        self.max_concentration = max_concentration
        self.max_turnover_daily = max_turnover_daily

    def rolling_drawdown(self, equity: pd.Series) -> pd.Series:
        peak = equity.expanding().max()
        return (peak - equity) / peak.replace(0, np.nan)

    def rolling_volatility(self, returns: pd.Series, window: int = 20) -> pd.Series:
        return returns.rolling(window, min_periods=window).std() * np.sqrt(252)

    def exposure(self, positions: pd.Series, prices: pd.Series) -> pd.Series:
        return positions * prices

    def beta_exposure(self, asset_returns: pd.Series, market_returns: pd.Series, window: int = 60) -> pd.Series:
        cov = asset_returns.rolling(window).cov(market_returns)
        var = market_returns.rolling(window).var()
        return cov / var.replace(0, np.nan)

    def concentration(self, weights: pd.Series) -> float:
        return float(weights.abs().max()) if len(weights) else 0.0

    def turnover(self, positions: pd.Series) -> float:
        return float(positions.diff().abs().sum() / max(len(positions), 1))

    def check_kill_switches(
        self,
        equity: pd.Series,
        returns: pd.Series,
        volume: pd.Series | None = None,
    ) -> dict[str, bool]:
        flags = {
            "drawdown_stop": False,
            "vol_spike_stop": False,
            "liquidity_stop": False,
        }
        dd = self.rolling_drawdown(equity)
        if not dd.empty and dd.iloc[-1] > self.max_drawdown:
            flags["drawdown_stop"] = True
        vol = self.rolling_volatility(returns)
        if len(vol.dropna()) >= 20:
            base = vol.dropna().iloc[-20:-1].mean()
            if base > 0 and vol.iloc[-1] > base * self.max_vol_spike:
                flags["vol_spike_stop"] = True
        if volume is not None and len(volume) >= 20:
            if volume.iloc[-1] < volume.iloc[-20:-1].median() * 0.3:
                flags["liquidity_stop"] = True
        return flags

    def report(self, equity: pd.Series, returns: pd.Series) -> dict:
        dd = self.rolling_drawdown(equity)
        return {
            "current_drawdown": float(dd.iloc[-1]) if len(dd) else 0.0,
            "rolling_vol": float(self.rolling_volatility(returns).iloc[-1])
            if len(returns) > 20
            else 0.0,
            "kill_switches": self.check_kill_switches(equity, returns),
        }
