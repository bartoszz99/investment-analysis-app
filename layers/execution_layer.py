"""Execution v2: lagged fills, spread, slippage, market impact, partial fills."""

import pandas as pd

FEE_RATE = 0.001
SLIPPAGE_RATE = 0.001
POSITION_SIZE = 0.95
START_CAPITAL = 10000
DEFAULT_SPREAD_BPS = 5.0
MAX_ADV_PARTICIPATION = 0.05


def execution_price(row: pd.Series) -> float:
    if "Open" in row.index and pd.notna(row["Open"]):
        return float(row["Open"])
    return float(row["Close"])


def lag_signal_to_execution(position: pd.Series) -> pd.Series:
    """exec_position[t] = signal from t-1; trade at Open[t]."""
    return position.shift(1).fillna(0).astype(int)


def estimate_market_impact(
    volatility: float,
    participation_rate: float,
    volume: float,
    notional: float,
    impact_coef: float = 0.1,
) -> float:
    """
    Simplified square-root impact model (bps).
    participation_rate = notional / (price * ADV)
    """
    if volume <= 0 or notional <= 0 or volatility <= 0:
        return 0.0
    part = min(participation_rate, MAX_ADV_PARTICIPATION)
    return impact_coef * volatility * (part**0.5) * 10000


def effective_execution_price(
    base_price: float,
    side: str,
    spread_bps: float = DEFAULT_SPREAD_BPS,
    slippage_rate: float = SLIPPAGE_RATE,
    impact_bps: float = 0.0,
) -> float:
    spread = spread_bps / 10000.0
    impact = impact_bps / 10000.0
    if side == "buy":
        return base_price * (1 + spread / 2 + slippage_rate + impact)
    return base_price * (1 - spread / 2 - slippage_rate - impact)


def apply_partial_fill(
    requested_shares: float,
    available_liquidity: float,
    participation_limit: float = MAX_ADV_PARTICIPATION,
) -> float:
    cap = available_liquidity * participation_limit
    return min(requested_shares, cap) if cap > 0 else 0.0


def buy(
    cash: float,
    price: float,
    volume: float = 0.0,
    volatility: float = 0.2,
    spread_bps: float = DEFAULT_SPREAD_BPS,
) -> tuple[float, float, int]:
    invest = cash * POSITION_SIZE
    part = invest / max(price * max(volume, 1), 1)
    impact_bps = estimate_market_impact(volatility, part, volume, invest)
    exec_p = effective_execution_price(price, "buy", spread_bps, SLIPPAGE_RATE, impact_bps)
    fee = invest * FEE_RATE
    shares_req = (invest - fee) / exec_p
    shares = apply_partial_fill(shares_req, volume) if volume > 0 else shares_req
    return cash - invest, shares, 1 if shares > 0 else 0


def sell(
    cash: float,
    shares: float,
    price: float,
    volume: float = 0.0,
    volatility: float = 0.2,
    spread_bps: float = DEFAULT_SPREAD_BPS,
) -> tuple[float, float, int]:
    notional = shares * price
    part = notional / max(price * max(volume, 1), 1)
    impact_bps = estimate_market_impact(volatility, part, volume, notional)
    exec_p = effective_execution_price(price, "sell", spread_bps, SLIPPAGE_RATE, impact_bps)
    sell_shares = apply_partial_fill(shares, volume) if volume > 0 else shares
    proceeds = sell_shares * exec_p
    fee = proceeds * FEE_RATE
    return cash + proceeds - fee, shares - sell_shares, 1 if sell_shares > 0 else 0


def turnover_cost(prev_pos: int, new_pos: int, notional: float) -> float:
    if prev_pos == new_pos:
        return 0.0
    return notional * FEE_RATE
