import pandas as pd

WEIGHT_RETURN = 0.40
WEIGHT_SHARPE = 0.30
WEIGHT_DRAWDOWN = 0.20
WEIGHT_STABILITY = 0.10

REQUIRED_COLUMNS = ("return_pct", "sharpe", "max_drawdown")


def _validate_results_df(results_df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in results_df.columns]
    if missing:
        raise ValueError(f"Brak kolumn w results_df: {missing}")


def _raw_metrics(results_df: pd.DataFrame) -> dict[str, float]:
    _validate_results_df(results_df)
    std_return = results_df["return_pct"].std()
    std_sharpe = results_df["sharpe"].std()
    if pd.isna(std_return):
        std_return = 0.0
    if pd.isna(std_sharpe):
        std_sharpe = 0.0
    return {
        "return": float(results_df["return_pct"].mean()),
        "sharpe": float(results_df["sharpe"].mean()),
        "drawdown": float(results_df["max_drawdown"].mean()),
        "stability": float(std_return + std_sharpe),
    }


def _normalize_values(values: list[float], higher_is_better: bool) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if high == low:
        return [0.5] * len(values)
    if higher_is_better:
        return [(v - low) / (high - low) for v in values]
    return [(high - v) / (high - low) for v in values]


def _normalize_across_strategies(raw_by_strategy: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    names = list(raw_by_strategy.keys())
    norm: dict[str, dict[str, float]] = {name: {} for name in names}

    for key, higher_is_better in (
        ("return", True),
        ("sharpe", True),
        ("drawdown", False),
        ("stability", False),
    ):
        values = [raw_by_strategy[name][key] for name in names]
        scaled = _normalize_values(values, higher_is_better)
        for name, value in zip(names, scaled):
            norm[name][key] = value

    return norm


def score_strategy(
    results_df: pd.DataFrame,
    normalized: dict[str, float] | None = None,
) -> dict:
    """
    Liczy score 0-1 z znormalizowanych skladowych.
    Bez normalized — uzyj rank_strategies() do porownania miedzy strategiami.
    """
    raw = _raw_metrics(results_df)
    if normalized is None:
        normalized = {key: 0.5 for key in ("return", "sharpe", "drawdown", "stability")}

    score = (
        WEIGHT_RETURN * normalized["return"]
        + WEIGHT_SHARPE * normalized["sharpe"]
        + WEIGHT_DRAWDOWN * normalized["drawdown"]
        + WEIGHT_STABILITY * normalized["stability"]
    )

    return {
        "score": score,
        "return": raw["return"],
        "sharpe": raw["sharpe"],
        "drawdown": raw["drawdown"],
        "stability": raw["stability"],
        "normalized": normalized,
    }


def rank_strategies(results_dict: dict[str, pd.DataFrame]) -> list[dict]:
    """results_dict: nazwa strategii -> DataFrame (walk-forward lub robustness)."""
    valid = {name: df for name, df in results_dict.items() if df is not None and not df.empty}
    if not valid:
        return []

    raw_by_strategy = {name: _raw_metrics(df) for name, df in valid.items()}
    norm_by_strategy = _normalize_across_strategies(raw_by_strategy)

    ranked = []
    for name, df in valid.items():
        entry = score_strategy(df, norm_by_strategy[name])
        entry["strategy"] = name
        ranked.append(entry)

    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked


def results_from_robustness(all_results: list[dict]) -> dict[str, pd.DataFrame]:
    """Konwertuje output run_robustness_test() na format results_dict."""
    if not all_results:
        return {}

    strategy_names = list(all_results[0]["results"].keys())
    out: dict[str, pd.DataFrame] = {}

    for name in strategy_names:
        rows = []
        for item in all_results:
            metrics = item["results"][name]
            rows.append(
                {
                    "ticker": item["ticker"],
                    "return_pct": metrics["return_pct"],
                    "sharpe": metrics["sharpe"],
                    "max_drawdown": metrics["max_drawdown"],
                }
            )
        out[name] = pd.DataFrame(rows)

    return out


def results_from_walk_forward(walk_forward_results: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Walk-forward juz zwraca results_dict — przekaz bez zmian."""
    return walk_forward_results


def print_ranking_table(ranked: list[dict], title: str = "Ranking strategii") -> None:
    print(f"\n=== {title} ===")
    if not ranked:
        print("Brak wynikow do rankingu.")
        return

    header = (
        f"{'#':<4} {'Strategy':<16} {'Score':>8} {'Return%':>10} "
        f"{'Sharpe':>8} {'Drawdown%':>11} {'Stability':>10}"
    )
    print(header)
    print("-" * len(header))

    for rank, entry in enumerate(ranked, start=1):
        print(
            f"{rank:<4} {entry['strategy']:<16} {entry['score']:>8.4f} "
            f"{entry['return']:>9.2f}% {entry['sharpe']:>8.4f} "
            f"{entry['drawdown']:>10.2f}% {entry['stability']:>10.4f}"
        )

    print(f"\nNajlepsza strategia: {ranked[0]['strategy']} (score={ranked[0]['score']:.4f})")
    if len(ranked) > 1:
        print(
            f"Najgorsza strategia: {ranked[-1]['strategy']} "
            f"(score={ranked[-1]['score']:.4f})"
        )


def rank_and_print(
    results_dict: dict[str, pd.DataFrame],
    title: str = "Ranking strategii",
) -> list[dict]:
    ranked = rank_strategies(results_dict)
    print_ranking_table(ranked, title)
    return ranked


if __name__ == "__main__":
    import config
    from data_loader import clean_data, fetch_data
    from walk_forward import DEFAULT_STRATEGIES, run_walk_forward_multi
    from robustness import run_robustness_test

    print("--- Ranking: walk-forward ---")
    raw = fetch_data(config.TICKER, "1y")
    base_df = clean_data(raw)
    wf = run_walk_forward_multi(base_df, DEFAULT_STRATEGIES, train_size=60, test_size=20)
    rank_and_print(wf, "Ranking (walk-forward)")

    print("\n--- Ranking: robustness ---")
    rob = run_robustness_test()
    rank_and_print(results_from_robustness(rob), "Ranking (robustness)")
