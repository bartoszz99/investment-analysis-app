"""
Backtest engine: vectorized (compat) + event-driven simulation.
Temporal: signal[t] -> fill Open[t+1], MTM Close[t].
"""

from dataclasses import dataclass, field
from enum import Enum, auto

import pandas as pd

from layers.execution_layer import (
    FEE_RATE,
    POSITION_SIZE,
    START_CAPITAL as _DEFAULT_CAPITAL,
    buy,
    execution_price,
    lag_signal_to_execution,
    sell,
)
from layers.microstructure import build_microstructure_features, execution_context

START_CAPITAL = _DEFAULT_CAPITAL


class EventType(Enum):
    MARKET_OPEN = auto()
    SIGNAL = auto()
    ORDER_SUBMIT = auto()
    FILL = auto()
    MARKET_CLOSE = auto()


@dataclass
class Event:
    day_index: int
    event_type: EventType
    payload: dict = field(default_factory=dict)


def _daily_returns(equity: list[float]) -> list[float]:
    out = []
    for i in range(1, len(equity)):
        if equity[i - 1] > 0:
            out.append(equity[i] / equity[i - 1] - 1)
    return out


def compute_metrics(equity: list[float], start_capital: float, trades: int) -> dict:
    final = equity[-1]
    ret_pct = (final / start_capital - 1) * 100
    peak = equity[0]
    max_dd = 0.0
    for v in equity:
        peak = max(peak, v)
        if peak > 0:
            max_dd = max(max_dd, (peak - v) / peak)
    daily = _daily_returns(equity)
    sharpe = 0.0
    if len(daily) > 1:
        mu = sum(daily) / len(daily)
        var = sum((r - mu) ** 2 for r in daily) / len(daily)
        sharpe = mu / (var**0.5) if var > 0 else 0.0
    return {
        "equity": equity,
        "final_capital": final,
        "return_pct": ret_pct,
        "max_drawdown": max_dd * 100,
        "sharpe": sharpe,
        "trades": trades,
    }


def run_backtest_event_driven(
    df: pd.DataFrame,
    start_capital: float = START_CAPITAL,
    execution_latency: int = 0,
    use_microstructure: bool = True,
) -> dict:
    """
    Event queue per bar:
    MARKET_OPEN -> SIGNAL (prior day) -> ORDER_SUBMIT -> FILL @ Open -> MARKET_CLOSE MTM.
    execution_latency: extra bars before fill (0 = next open as in vectorized).
    """
    if "position" not in df.columns:
        raise ValueError("Missing position")

    data = build_microstructure_features(df) if use_microstructure else df.copy()
    exec_pos = lag_signal_to_execution(data["position"])
    if execution_latency > 0:
        exec_pos = exec_pos.shift(execution_latency).fillna(0).astype(int)

    cash = float(start_capital)
    shares = 0.0
    equity = []
    trades = 0
    prev_pos = 0
    pending_target: int | None = None

    for i in range(len(data)):
        row = data.iloc[i]
        close = float(row["Close"])
        ctx = execution_context(row) if use_microstructure else {}
        vol = ctx.get("volatility", 0.2)
        volume = ctx.get("volume", 1e6)
        spread_bps = ctx.get("spread_bps", 5.0)

        # FILL at open from signal known previous close
        target = int(exec_pos.iloc[i])
        if pending_target is not None:
            target = pending_target
            pending_target = None

        px = execution_price(row)
        if target == 1 and prev_pos == 0:
            cash, shares, n = buy(cash, px, volume=volume, volatility=vol, spread_bps=spread_bps)
            trades += n
        elif target == 0 and prev_pos == 1:
            cash, shares, n = sell(cash, shares, px, volume=volume, volatility=vol, spread_bps=spread_bps)
            trades += n

        prev_pos = target
        equity.append(cash + shares * close)

    return compute_metrics(equity, start_capital, trades)


def run_backtest(
    df: pd.DataFrame,
    start_capital: float = START_CAPITAL,
    event_driven: bool = False,
) -> dict:
    """Default: vectorized causal loop; set event_driven=True for microstructure-aware sim."""
    if event_driven:
        return run_backtest_event_driven(df, start_capital)

    exec_pos = lag_signal_to_execution(df["position"])
    cash = float(start_capital)
    shares = 0.0
    equity = []
    trades = 0
    prev = 0

    for i in range(len(df)):
        row = df.iloc[i]
        close = float(row["Close"])
        target = int(exec_pos.iloc[i])
        px = execution_price(row)
        if target == 1 and prev == 0:
            cash, shares, n = buy(cash, px)
            trades += n
        elif target == 0 and prev == 1:
            cash, shares, n = sell(cash, shares, px)
            trades += n
        prev = target
        equity.append(cash + shares * close)

    return compute_metrics(equity, start_capital, trades)
