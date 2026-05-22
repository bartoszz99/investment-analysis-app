"""
Institutional research stack entry point (compatible with main.py).
"""

import pandas as pd

import config
from audit.leakage_audit import generate_audit_report
from audit.sanity import print_sanity_results, run_institutional_sanity
from data_loader import load_market_data
from layers.alpha_layer import AlphaModel
from layers.backtest_engine import START_CAPITAL, run_backtest
from layers.feature_store import FeatureStore
from layers.risk_engine import RiskEngine
from ml.train_pipeline import MLTrainPipeline
from research.experiment_registry import ExperimentRegistry
from strategies import DEFAULT_STRATEGIES, BuyAndHold, SmaCrossover
from validation.stress_tests import run_stress_suite


def run_research_stack(period: str | None = None) -> dict:
    period = period or config.PERIOD
    df = load_market_data(config.TICKER, period)

    store = FeatureStore()
    features = store.build(df)
    print(f"\n=== Feature Store ({len(store.list_features())} features) ===")
    print(features[store.list_features()].tail(3))

    alpha = AlphaModel(store)
    alpha_df = alpha.forecast(df, dynamic_weights=True)
    alpha_metrics = run_backtest(alpha_df, START_CAPITAL)
    print(f"\n=== Alpha ensemble ===")
    print(f"Return: {alpha_metrics['return_pct']:+.2f}% Sharpe: {alpha_metrics['sharpe']:.4f}")

    classic = {s.name: run_backtest(s.apply(df.copy()), START_CAPITAL) for s in DEFAULT_STRATEGIES}
    generate_audit_report(df, DEFAULT_STRATEGIES, classic)

    if len(df) >= 150:
        ml = MLTrainPipeline()
        ml_result = ml.rolling_oos_train(df, "research_ml")
        print(f"\n=== ML OOS (purged rolling) ===")
        print(f"Sharpe: {ml_result.oos_sharpe:.4f} RMSE: {ml_result.diagnostics.get('rmse', 0):.4f}")
        print(f"Top features: {sorted(ml_result.feature_importance.items(), key=lambda x: -x[1])[:3]}")
    else:
        print("\n[SKIP] ML pipeline needs >= 150 bars")
        ml_result = None

    rets = pd.Series(alpha_metrics["equity"]).pct_change().dropna()
    stress = run_stress_suite(rets)
    print(f"\n=== Validation ===")
    print(f"Monte Carlo: {stress['monte_carlo']['warning']} (stability={stress['monte_carlo']['stability_score']:.2f})")

    eq = pd.Series(alpha_metrics["equity"])
    risk = RiskEngine().report(eq, rets)
    print(f"\n=== Risk === {risk}")

    registry = ExperimentRegistry()
    registry.register(
        features=store.list_features(),
        model_params={"alpha_weights": alpha.weights},
        train_window=120,
        test_window=20,
        metrics=alpha_metrics,
        sharpe=alpha_metrics["sharpe"],
        drawdown=alpha_metrics["max_drawdown"],
        leakage_score=0.0,
        notes=f"{config.TICKER} {period}",
    )

    print_sanity_results(run_institutional_sanity(df, {**classic, "Alpha": alpha_metrics}, DEFAULT_STRATEGIES))
    return {"alpha": alpha_metrics, "classic": classic, "ml": ml_result, "stress": stress}


if __name__ == "__main__":
    run_research_stack("1y")
