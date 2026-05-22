import pandas as pd

from layers.backtest_engine import START_CAPITAL
from layers.purged_walk_forward import purged_walk_forward
from strategies import BaseStrategy, BuyAndHold, SmaCrossover

DEFAULT_STRATEGIES: list[BaseStrategy] = [
    SmaCrossover(),
    BuyAndHold(),
]


def run_walk_forward(
    df: pd.DataFrame,
    strategy: BaseStrategy,
    train_size: int,
    test_size: int,
    start_capital: float = START_CAPITAL,
    embargo: int = 0,
) -> pd.DataFrame:
    return purged_walk_forward(
        df, strategy, train_size, test_size, embargo, start_capital
    )


def run_walk_forward_multi(
    df: pd.DataFrame,
    strategies: list[BaseStrategy],
    train_size: int,
    test_size: int,
    start_capital: float = START_CAPITAL,
    embargo: int = 0,
) -> dict[str, pd.DataFrame]:
    return {
        s.name: run_walk_forward(df, s, train_size, test_size, start_capital, embargo)
        for s in strategies
    }


def aggregate_results(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame(columns=["metric", "return_pct", "sharpe", "max_drawdown"])
    return pd.DataFrame(
        [
            {"metric": "mean", "return_pct": results["return_pct"].mean(), "sharpe": results["sharpe"].mean(), "max_drawdown": results["max_drawdown"].mean()},
            {"metric": "std", "return_pct": results["return_pct"].std(), "sharpe": results["sharpe"].std(), "max_drawdown": results["max_drawdown"].std()},
            {"metric": "min", "return_pct": results["return_pct"].min(), "sharpe": results["sharpe"].min(), "max_drawdown": results["max_drawdown"].min()},
            {"metric": "max", "return_pct": results["return_pct"].max(), "sharpe": results["sharpe"].max(), "max_drawdown": results["max_drawdown"].max()},
        ]
    )


def print_walk_forward_report(results: pd.DataFrame, strategy_name: str, train_size: int, test_size: int) -> None:
    print(f"\n=== Purged walk-forward: {strategy_name} (train={train_size}, test={test_size}) ===")
    if results.empty:
        print("Brak okien.")
        return
    print(results.to_string(index=False))
    print("\n--- Agregaty ---")
    print(aggregate_results(results).to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def run_walk_forward_analysis(
    df: pd.DataFrame,
    strategies: list[BaseStrategy] | None = None,
    train_size: int = 60,
    test_size: int = 20,
    embargo: int = 1,
) -> dict[str, pd.DataFrame]:
    strategies = strategies or DEFAULT_STRATEGIES
    all_results = run_walk_forward_multi(df, strategies, train_size, test_size, embargo=embargo)
    for s in strategies:
        print_walk_forward_report(all_results[s.name], s.name, train_size, test_size)
    return all_results


if __name__ == "__main__":
    import config
    from data_loader import load_market_data

    run_walk_forward_analysis(load_market_data(config.TICKER, "1y"), embargo=1)
