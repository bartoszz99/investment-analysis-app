import pandas as pd

from layers.backtest_engine import START_CAPITAL, run_backtest
from layers.purged_walk_forward import calibration_test_split, purged_walk_forward
from layers.regime_layer import ALL_REGIMES, detect_regime
from strategy_regime_map import DEFAULT_REGIME_MAP, apply_with_regime_filter, FILTER_HARD
from strategies.base import BaseStrategy

WF_TRAIN_SIZE = 60
WF_TEST_SIZE = 20
WF_N_WINDOWS = 5
DRAWDOWN_PENALTY = 0.5
CALIBRATION_RATIO = 0.7


def _walk_forward_scores(
    df: pd.DataFrame,
    strategy: BaseStrategy,
    train_size: int = WF_TRAIN_SIZE,
    test_size: int = WF_TEST_SIZE,
    n_windows: int = WF_N_WINDOWS,
    embargo: int = 1,
) -> dict[str, float]:
    wf = purged_walk_forward(df, strategy, train_size, test_size, embargo)
    if wf.empty:
        return {"sharpe": 0.0, "max_drawdown": 0.0}
    recent = wf.tail(n_windows)
    return {
        "sharpe": float(recent["sharpe"].mean()),
        "max_drawdown": float(recent["max_drawdown"].mean()),
    }


def _strategy_score(sharpe: float, max_drawdown: float) -> float:
    sharpe_part = max(sharpe, 0.0)
    dd_penalty = max(1.0 - DRAWDOWN_PENALTY * (max_drawdown / 100.0), 0.1)
    return sharpe_part * dd_penalty


def compute_strategy_weights(
    df: pd.DataFrame,
    strategies: list[BaseStrategy],
    regime_map: dict[str, list[str]] | None = None,
    train_size: int = WF_TRAIN_SIZE,
    test_size: int = WF_TEST_SIZE,
    n_windows: int = WF_N_WINDOWS,
    embargo: int = 1,
) -> tuple[dict[str, dict[str, float]], pd.DataFrame]:
    regime_map = regime_map or DEFAULT_REGIME_MAP
    base = detect_regime(df.copy())
    cal_df, _ = calibration_test_split(base, CALIBRATION_RATIO)
    cal_len = len(cal_df)

    performance = {
        s.name: _walk_forward_scores(cal_df, s, train_size, test_size, n_windows, embargo)
        for s in strategies
    }

    weights_by_regime: dict[str, dict[str, float]] = {}
    for regime in ALL_REGIMES:
        active = [s for s in strategies if regime in regime_map.get(s.name, list(ALL_REGIMES))]
        if not active:
            weights_by_regime[regime] = {}
            continue
        base_w = 1.0 / len(active)
        raw = {
            s.name: base_w * _strategy_score(performance[s.name]["sharpe"], performance[s.name]["max_drawdown"])
            for s in active
        }
        total = sum(raw.values())
        weights_by_regime[regime] = (
            {n: v / total for n, v in raw.items()} if total > 0 else {s.name: base_w for s in active}
        )

    equal_w = 1.0 / len(strategies) if strategies else 0.0
    weight_rows = []
    for j, idx in enumerate(base.index):
        row = {s.name: equal_w for s in strategies} if j < cal_len else {s.name: 0.0 for s in strategies}
        if j >= cal_len and pd.notna(base.loc[idx, "regime"]):
            regime = base.loc[idx, "regime"]
            if regime in weights_by_regime:
                row = {s.name: 0.0 for s in strategies}
                for name, w in weights_by_regime[regime].items():
                    row[name] = w
        weight_rows.append(row)

    return weights_by_regime, pd.DataFrame(weight_rows, index=base.index)


def prepare_strategy_signals(
    base_df: pd.DataFrame,
    strategies: list[BaseStrategy],
    regime_map: dict[str, list[str]] | None = None,
) -> dict[str, pd.DataFrame]:
    regime_map = regime_map or DEFAULT_REGIME_MAP
    return {
        s.name: apply_with_regime_filter(base_df, s, regime_map, FILTER_HARD) for s in strategies
    }


def allocate_capital(
    df: pd.DataFrame,
    strategy_signals: dict[str, pd.DataFrame],
    weights: pd.DataFrame,
    start_capital: float = START_CAPITAL,
) -> dict:
    index = df.index
    names = list(strategy_signals.keys())
    strategy_equity = {}
    strategy_returns = {}
    for name in names:
        metrics = run_backtest(strategy_signals[name], start_capital)
        strategy_equity[name] = metrics["equity"]
        rets = [0.0]
        for i in range(1, len(metrics["equity"])):
            p = metrics["equity"][i - 1]
            rets.append((metrics["equity"][i] / p - 1.0) if p > 0 else 0.0)
        strategy_returns[name] = rets

    sleeves = {n: 0.0 for n in names}
    portfolio_equity = []
    contribution_rows = []
    for i in range(len(index)):
        w = {n: float(weights.iloc[i][n]) if n in weights.columns else 0.0 for n in names}
        ws = sum(w.values())
        if i == 0:
            total = start_capital
        else:
            for n in names:
                sleeves[n] *= 1.0 + strategy_returns[n][i]
            total = sum(sleeves.values())
        if ws > 0:
            for n in names:
                sleeves[n] = total * (w[n] / ws)
        elif i == 0:
            for n in names:
                sleeves[n] = start_capital / len(names)
        portfolio_equity.append(sum(sleeves.values()))
        contribution_rows.append(dict(sleeves))

    return {
        "equity": portfolio_equity,
        "strategy_equity": strategy_equity,
        "contribution": pd.DataFrame(contribution_rows, index=index),
        "weights": weights,
    }


def run_portfolio_allocation(
    base_df: pd.DataFrame,
    strategies: list[BaseStrategy],
    regime_map: dict[str, list[str]] | None = None,
    start_capital: float = START_CAPITAL,
) -> dict:
    base = detect_regime(base_df.copy())
    w_reg, w_time = compute_strategy_weights(base, strategies, regime_map)
    signals = prepare_strategy_signals(base, strategies, regime_map)
    out = allocate_capital(base, signals, w_time, start_capital)
    out["weights_by_regime"] = w_reg
    return out
