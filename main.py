import matplotlib.pyplot as plt

import config
from audit.leakage_audit import generate_audit_report, run_leakage_audit
from audit.sanity import print_sanity_results, run_institutional_sanity
from data_loader import load_market_data
from layers.backtest_engine import START_CAPITAL, run_backtest
from layers.execution_layer import FEE_RATE, POSITION_SIZE, SLIPPAGE_RATE
from plotter import plot_price_with_sma
from strategies import BaseStrategy, BuyAndHold, SmaCrossover

DEFAULT_STRATEGIES: list[BaseStrategy] = [
    SmaCrossover(),
    BuyAndHold(),
]


def print_stats(df, ticker: str) -> None:
    close = df["Close"]
    print(f"\n=== Statystyki: {ticker} ===")
    print(f"Liczba sesji:     {len(df)}")
    print(f"Zakres dat:       {df.index.min().date()} — {df.index.max().date()}")
    print(f"Cena min:         {close.min():.2f}")
    print(f"Cena max:         {close.max():.2f}")
    print(f"Cena srednia:     {close.mean():.2f}")
    print(f"Ostatnia cena:    {close.iloc[-1]:.2f}")


def run_backtests(base_df, strategies: list[BaseStrategy], start_capital: float = START_CAPITAL):
    return {s.name: run_backtest(s.apply(base_df.copy()), start_capital) for s in strategies}


def run_backtest_sanity_checks(df, trades: dict | None = None, strategies=None) -> bool:
    from audit.sanity import run_institutional_sanity

    strategies = strategies or DEFAULT_STRATEGIES
    trades = trades or {}
    results = {n: {"trades": trades.get(n, 0), "return_pct": 0, "sharpe": 0} for n in trades}
    for s in strategies:
        if s.name not in results:
            results[s.name] = run_backtest(s.apply(df.copy()))
    return print_sanity_results(run_institutional_sanity(df, results, strategies))


def print_backtest_metrics(name: str, metrics: dict) -> None:
    print(f"\n--- {name} ---")
    print(f"Kapital koncowy:   {metrics['final_capital']:,.2f} USD")
    print(f"Zwrot:             {metrics['return_pct']:+.2f}%")
    print(f"Max drawdown:      {metrics['max_drawdown']:.2f}%")
    print(f"Sharpe:            {metrics['sharpe']:.4f}")
    print(f"Liczba transakcji: {metrics['trades']}")


def print_comparison(results: dict, benchmark: str = "Buy & Hold") -> None:
    if benchmark not in results:
        return
    bench = results[benchmark]
    print(f"\n=== Porownanie (vs {benchmark}) ===")
    for name, m in results.items():
        if name == benchmark:
            continue
        print(
            f"{name}: {m['return_pct']:+.2f}% (diff {m['return_pct']-bench['return_pct']:+.2f} p.p.), "
            f"Sharpe {m['sharpe']:.4f}"
        )


def plot_equity_comparison(df, ticker: str, results: dict, save_path: str = "equity_comparison.png") -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, m in results.items():
        ax.plot(df.index, m["equity"], label=name, linewidth=1.5)
    ax.axhline(START_CAPITAL, color="gray", linestyle="--", linewidth=1)
    ax.set_title(f"{ticker} — equity (causal backtest)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"\nWykres: {save_path}")
    if config.SHOW_PLOT:
        plt.show()
    else:
        plt.close(fig)


def run_pipeline(strategies: list[BaseStrategy] | None = None) -> dict:
    strategies = strategies or DEFAULT_STRATEGIES

    base_df = load_market_data(config.TICKER, config.PERIOD)
    print_stats(base_df, config.TICKER)

    print("\n=== LEAKAGE AUDIT (pre-backtest) ===")
    run_leakage_audit(base_df, "market_data")
    primary = strategies[0]
    preview = primary.apply(base_df.copy())
    print(f"\nOstatnie wiersze ({primary.name}):\n{preview.tail()}")

    print(
        f"\n=== Backtest [{FEE_RATE*100:.1f}% fee, {SLIPPAGE_RATE*100:.1f}% slip, exec=Open t+1] ==="
    )
    results = run_backtests(base_df, strategies)
    generate_audit_report(base_df, strategies, results)
    run_backtest_sanity_checks(base_df, {n: m["trades"] for n, m in results.items()}, strategies)

    for name, m in results.items():
        print_backtest_metrics(name, m)
    print_comparison(results)
    plot_equity_comparison(base_df, config.TICKER, results)

    if isinstance(primary, SmaCrossover):
        plot_price_with_sma(
            preview, config.TICKER, primary.sma_short, primary.sma_long,
            save_path=config.CHART_FILE, show=config.SHOW_PLOT,
        )
    return results


# Backward compatibility
run_backtest_buy_and_hold = lambda df, sc=START_CAPITAL: run_backtest(BuyAndHold().apply(df), sc)
run_backtest_sma = run_backtest


if __name__ == "__main__":
    run_pipeline()
