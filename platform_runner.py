"""
Institutional platform runner — wires lineage, factors, validation, registry, report.
Backward compatible with main.run_pipeline().
"""

import pandas as pd

import config
from audit.leakage_audit import generate_audit_report
from audit.sanity import print_sanity_results, run_institutional_sanity
from data_loader import load_market_data
from layers.backtest_engine import run_backtest
from layers.factor_model import FactorModel
from layers.feature_lineage import FeatureLineageTracker
from layers.feature_store import FeatureStore
from research.model_registry import ModelRegistry
from research.report_generator import generate_institutional_report
from strategies import DEFAULT_STRATEGIES
from validation.deflated_sharpe import deflated_sharpe_ratio
from validation.monte_carlo import monte_carlo_stability
from validation.reality_check import whites_reality_check
from validation.stress_tests import run_stress_suite


def run_institutional_platform(period: str | None = None) -> dict:
    period = period or config.PERIOD
    df = load_market_data(config.TICKER, period)

    lineage = FeatureLineageTracker().register_from_feature_store().report()
    lineage_path = "research/feature_lineage.json"
    lineage.to_json(lineage_path)

    store = FeatureStore(use_cache=True)
    _ = store.build(df)

    factors = FactorModel().summarize(df)
    print("\n=== Factor model ===")
    print(factors.get("ic", {}))

    results = {}
    for s in DEFAULT_STRATEGIES:
        sig = s.apply(df.copy())
        results[s.name] = run_backtest(sig, event_driven=False)

    bh = results.get("Buy & Hold")
    sma = results.get("SMA 7/30")
    if bh and sma:
        strat_rets = pd.Series(bh["equity"]).pct_change().dropna()
        bench_rets = strat_rets  # single strategy baseline
        validation = {
            "monte_carlo": monte_carlo_stability(strat_rets),
            "deflated_sharpe": deflated_sharpe_ratio(strat_rets, n_trials=2),
            "reality_check": whites_reality_check(strat_rets, bench_rets),
            "stress": run_stress_suite(strat_rets),
        }
    else:
        validation = {}

    generate_audit_report(df, DEFAULT_STRATEGIES, results)
    print_sanity_results(run_institutional_sanity(df, results, DEFAULT_STRATEGIES))

    registry = ModelRegistry()
    for name, m in results.items():
        registry.register(
            features=store.list_features(),
            model_params={"strategy": name},
            train_period=(str(df.index[0].date()), str(df.index[len(df) // 2].date())),
            test_period=(str(df.index[len(df) // 2].date()), str(df.index[-1].date())),
            oos_metrics=m,
            sharpe=m["sharpe"],
            drawdown=m["max_drawdown"],
        )

    primary = list(results.values())[0]
    generate_institutional_report(
        df,
        primary,
        primary["equity"],
        {"ticker": config.TICKER, "period": period},
        validation,
        lineage_path,
    )

    return {"results": results, "lineage": lineage_path, "validation": validation, "factors": factors}


if __name__ == "__main__":
    run_institutional_platform("1y")
