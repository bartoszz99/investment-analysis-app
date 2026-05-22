import pandas as pd

from layers.regime_layer import ALL_REGIMES, detect_regime
from strategies.base import BaseStrategy

FILTER_NONE = "no regime filtering"
FILTER_HARD = "hard filter"

DEFAULT_REGIME_MAP: dict[str, list[str]] = {
    "SMA 7/30": [REGIME_TREND],
    "Buy & Hold": [REGIME_TREND, REGIME_MEAN_REVERSION],
}


def filter_by_regime(
    df: pd.DataFrame,
    strategy: BaseStrategy,
    regime_map: dict[str, list[str]] | None = None,
    mode: str = FILTER_HARD,
) -> pd.DataFrame:
    """
    Wylacza sygnaly (position=0) poza dozwolonymi rezimami.
    Wymaga kolumn: position, regime (lub zostanie dodana przez detect_regime).
    """
    if mode == FILTER_NONE:
        return df.copy()

    regime_map = regime_map or DEFAULT_REGIME_MAP
    result = df.copy()

    if "regime" not in result.columns:
        result = detect_regime(result)
    if "position" not in result.columns:
        raise ValueError("DataFrame musi zawierac kolumne position (najpierw strategy.apply)")

    allowed = regime_map.get(strategy.name, list(ALL_REGIMES))
    mask_allowed = result["regime"].isin(allowed)
    result.loc[~mask_allowed, "position"] = 0
    return result


def apply_with_regime_filter(
    base_df: pd.DataFrame,
    strategy: BaseStrategy,
    regime_map: dict[str, list[str]] | None = None,
    mode: str = FILTER_HARD,
) -> pd.DataFrame:
    """Preprocessing przed backtestem: sygnaly strategii + opcjonalny filtr rezimow."""
    regime_map = regime_map or DEFAULT_REGIME_MAP

    if mode == FILTER_NONE:
        return strategy.apply(base_df.copy())

    df = base_df.copy()
    if "regime" not in df.columns:
        df = detect_regime(df)

    signaled = strategy.apply(df.copy())
    return filter_by_regime(signaled, strategy, regime_map, mode)


def _mask_positions_for_regime(df: pd.DataFrame, regime_name: str) -> pd.DataFrame:
    result = df.copy()
    in_regime = result["regime"] == regime_name
    result["position"] = result["position"].where(in_regime, 0).astype(int)
    return result


def performance_per_regime(
    df: pd.DataFrame,
    strategy: BaseStrategy,
    start_capital: float,
) -> pd.DataFrame:
    """
    Backtest z pozycjami aktywnymi tylko w danym rezimie (pozostale dni: flat).
    Wymaga: Close, position, regime.
    """
    from main import run_backtest

    rows = []
    for regime_name in ALL_REGIMES:
        subset = df.dropna(subset=["regime"])
        if subset.empty or regime_name not in subset["regime"].values:
            rows.append(
                {
                    "regime": regime_name,
                    "return_pct": 0.0,
                    "sharpe": 0.0,
                    "max_drawdown": 0.0,
                    "trades": 0,
                    "sessions": 0,
                }
            )
            continue

        masked = _mask_positions_for_regime(subset, regime_name)
        metrics = run_backtest(masked, start_capital)
        sessions = int((subset["regime"] == regime_name).sum())
        rows.append(
            {
                "regime": regime_name,
                "return_pct": metrics["return_pct"],
                "sharpe": metrics["sharpe"],
                "max_drawdown": metrics["max_drawdown"],
                "trades": metrics["trades"],
                "sessions": sessions,
            }
        )

    return pd.DataFrame(rows)


def print_regime_performance_report(
    df: pd.DataFrame,
    strategy: BaseStrategy,
    start_capital: float,
    mode: str = FILTER_NONE,
) -> pd.DataFrame:
    print(f"\n=== Performance per regime: {strategy.name} (filtr: {mode}) ===")

    if "regime" not in df.columns:
        df = detect_regime(df)

    perf = performance_per_regime(df, strategy, start_capital)
    header = f"{'Regime':<18} {'Return%':>10} {'Sharpe':>8} {'MaxDD%':>10} {'Trades':>8} {'Sessions':>10}"
    print(header)
    print("-" * len(header))
    for _, row in perf.iterrows():
        print(
            f"{row['regime']:<18} {row['return_pct']:>9.2f}% "
            f"{row['sharpe']:>8.4f} {row['max_drawdown']:>9.2f}% "
            f"{int(row['trades']):>8} {int(row['sessions']):>10}"
        )
    return perf


def print_filter_summary(
    df: pd.DataFrame,
    strategy: BaseStrategy,
    regime_map: dict[str, list[str]] | None = None,
) -> None:
    regime_map = regime_map or DEFAULT_REGIME_MAP
    allowed = regime_map.get(strategy.name, list(ALL_REGIMES))
    valid = df.dropna(subset=["regime"])
    if valid.empty:
        return

    active = valid["position"].astype(bool)
    allowed_mask = valid["regime"].isin(allowed)
    blocked = active & ~allowed_mask

    print(f"\n--- Filtr rezimow: {strategy.name} ---")
    print(f"Dozwolone rezimy: {', '.join(allowed)}")
    print(f"Sesje z pozycja=1: {int(active.sum())}")
    print(f"Zablokowane sygnaly (hard filter): {int(blocked.sum())}")


if __name__ == "__main__":
    import config
    from data_loader import clean_data, fetch_data
    from main import START_CAPITAL, run_backtest
    from strategies import BuyAndHold, SmaCrossover

    raw = fetch_data(config.TICKER, "1y")
    base_df = clean_data(raw)

    for strategy in (SmaCrossover(), BuyAndHold()):
        print(f"\n{'=' * 50}")
        print(f"Strategia: {strategy.name}")

        unfiltered = apply_with_regime_filter(base_df, strategy, mode=FILTER_NONE)
        filtered = apply_with_regime_filter(base_df, strategy, mode=FILTER_HARD)

        m_raw = run_backtest(unfiltered, START_CAPITAL)
        m_filt = run_backtest(filtered, START_CAPITAL)

        print(f"\nBacktest bez filtra:  zwrot {m_raw['return_pct']:+.2f}%, Sharpe {m_raw['sharpe']:.4f}")
        print(f"Backtest hard filter: zwrot {m_filt['return_pct']:+.2f}%, Sharpe {m_filt['sharpe']:.4f}")

        print_filter_summary(filtered, strategy)
        print_regime_performance_report(unfiltered, strategy, START_CAPITAL, FILTER_NONE)
        print_regime_performance_report(filtered, strategy, START_CAPITAL, FILTER_HARD)
