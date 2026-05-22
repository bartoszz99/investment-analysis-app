import matplotlib.pyplot as plt

import config
from data_loader import clean_data, fetch_data
from main import START_CAPITAL, run_backtests
from strategies import BaseStrategy, BuyAndHold, SmaCrossover

TICKERS = ["AAPL", "MSFT", "SPY", "TSLA", "NVDA"]
ROBUSTNESS_PERIOD = "6mo"
CHART_FILE = "robustness_chart.png"
STABILITY_STD_THRESHOLD = 5.0

DEFAULT_STRATEGIES: list[BaseStrategy] = [
    SmaCrossover(),
    BuyAndHold(),
]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return (sum((v - m) ** 2 for v in values) / len(values)) ** 0.5


def _run_ticker(ticker: str, strategies: list[BaseStrategy]) -> dict:
    raw = fetch_data(ticker, ROBUSTNESS_PERIOD)
    base_df = clean_data(raw)
    backtest_results = run_backtests(base_df, strategies, START_CAPITAL)
    return {"ticker": ticker, "results": backtest_results}


def _print_results_table(rows: list[dict]) -> None:
    header = f"{'Ticker':<8} {'Strategy':<16} {'Return %':>10} {'Sharpe':>8} {'MaxDD %':>10}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['ticker']:<8} {row['strategy']:<16} "
            f"{row['return_pct']:>9.2f}% {row['sharpe']:>8.4f} {row['max_drawdown']:>9.2f}%"
        )


def _print_stability_summary(strategy_name: str, returns: list[float], sharpes: list[float]) -> None:
    mean_return = _mean(returns)
    std_return = _std(returns)
    mean_sharpe = _mean(sharpes)
    std_sharpe = _std(sharpes)

    print(f"\n=== Stabilnosc: {strategy_name} ({len(returns)} tickerow) ===")
    print(f"Sredni zwrot:              {mean_return:+.2f}%")
    print(f"Odchylenie std (zwrot):    {std_return:.2f} p.p.")
    print(f"Sredni Sharpe:             {mean_sharpe:.4f}")
    print(f"Odchylenie std (Sharpe):   {std_sharpe:.4f}")

    print("\n=== Komentarz stabilnosci ===")
    if std_return <= STABILITY_STD_THRESHOLD:
        print(
            f"Strategia {strategy_name} jest STABILNA: odchylenie zwrotow ({std_return:.2f} p.p.) "
            f"jest niskie wzgledem progu {STABILITY_STD_THRESHOLD:.1f} p.p."
        )
    else:
        print(
            f"Strategia {strategy_name} jest NIESTABILNA: duza roznica wynikow miedzy tickerami "
            f"(std zwrotow = {std_return:.2f} p.p.)."
        )


def _print_vs_benchmark(
    all_results: list[dict],
    strategies: list[BaseStrategy],
    benchmark_name: str = "Buy & Hold",
) -> None:
    for strategy in strategies:
        if strategy.name == benchmark_name:
            continue
        wins = sum(
            1
            for r in all_results
            if r["results"][strategy.name]["return_pct"]
            > r["results"][benchmark_name]["return_pct"]
        )
        print(f"{strategy.name} pokonala {benchmark_name} na {wins}/{len(all_results)} tickerach.")


def plot_robustness_chart(all_results: list[dict], strategies: list[BaseStrategy]) -> None:
    tickers = [r["ticker"] for r in all_results]
    n_strategies = len(strategies)
    x = list(range(len(tickers)))
    width = 0.8 / n_strategies

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, strategy in enumerate(strategies):
        offset = (i - n_strategies / 2 + 0.5) * width
        returns = [r["results"][strategy.name]["return_pct"] for r in all_results]
        ax.bar([xi + offset for xi in x], returns, width, label=strategy.name)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(tickers)
    ax.set_xlabel("Ticker")
    ax.set_ylabel("Return %")
    ax.set_title("Porownanie zwrotow strategii")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(CHART_FILE, dpi=150)
    print(f"\nWykres zapisany: {CHART_FILE}")
    if config.SHOW_PLOT:
        plt.show()
    else:
        plt.close(fig)


def run_robustness_test(strategies: list[BaseStrategy] | None = None) -> list[dict]:
    strategies = strategies or DEFAULT_STRATEGIES

    print(f"\n=== Robustness test (okres: {ROBUSTNESS_PERIOD}) ===")
    print(f"Tickery: {', '.join(TICKERS)}")
    print(f"Strategie: {', '.join(s.name for s in strategies)}")
    print(f"Kapital startowy: {START_CAPITAL:,.2f} USD\n")

    all_results = []
    table_rows = []

    for ticker in TICKERS:
        result = _run_ticker(ticker, strategies)
        all_results.append(result)
        for strategy in strategies:
            metrics = result["results"][strategy.name]
            table_rows.append(
                {
                    "ticker": ticker,
                    "strategy": strategy.name,
                    "return_pct": metrics["return_pct"],
                    "sharpe": metrics["sharpe"],
                    "max_drawdown": metrics["max_drawdown"],
                }
            )

    _print_results_table(table_rows)

    for strategy in strategies:
        if strategy.name == "Buy & Hold":
            continue
        returns = [r["results"][strategy.name]["return_pct"] for r in all_results]
        sharpes = [r["results"][strategy.name]["sharpe"] for r in all_results]
        _print_stability_summary(strategy.name, returns, sharpes)

    _print_vs_benchmark(all_results, strategies)
    plot_robustness_chart(all_results, strategies)

    return all_results


if __name__ == "__main__":
    run_robustness_test()
