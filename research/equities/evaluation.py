"""
Cross-sectional signal evaluation — IC, quintiles, regime, bootstrap.
Signal research only; no portfolio optimization.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.common.forward_returns import forward_return_panel
from research.common.signal_evaluation import (
    hit_ratio,
    quintile_spread,
    spearman_ic,
)
from research.equities.data_pipeline import stack_wide


HORIZONS_EQUITY = (5, 20)


def daily_cross_sectional_ic(signal: pd.DataFrame, forward_ret: pd.DataFrame) -> pd.Series:
    """Spearman IC per date via rank correlation (vectorized)."""
    common_cols = signal.columns.intersection(forward_ret.columns)
    s = signal[common_cols]
    f = forward_ret[common_cols]
    valid_count = s.notna() & f.notna()
    enough = valid_count.sum(axis=1) >= 10
    ic = s.rank(axis=1).corrwith(f.rank(axis=1), axis=1)
    return ic.loc[enough].dropna().rename("ic")


def mean_ic(signal: pd.DataFrame, forward_ret: pd.DataFrame) -> float:
    series = daily_cross_sectional_ic(signal, forward_ret)
    return float(series.mean()) if len(series) else np.nan


def pooled_quintile_spread(signal: pd.DataFrame, forward_ret: pd.DataFrame) -> dict:
    s = stack_wide(signal)
    f = stack_wide(forward_ret)
    return quintile_spread(s, f)


def pooled_hit_ratio(signal: pd.DataFrame, forward_ret: pd.DataFrame) -> float:
    s = stack_wide(signal)
    f = stack_wide(forward_ret)
    return hit_ratio(s, f)


def rolling_ic_series(
    signal: pd.DataFrame,
    forward_ret: pd.DataFrame,
    window: int = 60,
) -> pd.Series:
    daily = daily_cross_sectional_ic(signal, forward_ret)
    return daily.rolling(window, min_periods=max(20, window // 3)).mean()


def bootstrap_ic_stability(
    signal: pd.DataFrame,
    forward_ret: pd.DataFrame,
    *,
    n_draws: int = 100,
    seed: int = 42,
) -> dict:
    daily = daily_cross_sectional_ic(signal, forward_ret).dropna()
    if len(daily) < 30:
        return {"mean_ic": np.nan, "ic_std": np.nan, "survival_rate": np.nan, "p05": np.nan, "p95": np.nan}
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_draws):
        sample = daily.iloc[rng.integers(0, len(daily), size=len(daily))]
        means.append(float(sample.mean()))
    arr = np.array(means)
    return {
        "mean_ic": float(daily.mean()),
        "ic_std": float(arr.std()),
        "survival_rate": float((arr > 0).mean()) if daily.mean() > 0 else float((arr < 0).mean()),
        "p05": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
    }


def residual_sharpe_proxy(
    signal: pd.DataFrame,
    forward_ret: pd.DataFrame,
    *,
    sample_every: int = 5,
) -> float:
    """
    Long Q5 / short Q1 cross-sectional spread Sharpe (diagnostic, not tradable).
    """
    idx = signal.index.intersection(forward_ret.index)[::sample_every]
    spreads = []
    for dt in idx:
        s = signal.loc[dt].dropna()
        f = forward_ret.loc[dt].reindex(s.index).dropna()
        common = s.index.intersection(f.index)
        if len(common) < 15:
            continue
        qs = quintile_spread(s.loc[common], f.loc[common])
        sp = qs.get("spread_q5_q1", np.nan)
        if sp == sp:
            spreads.append(sp)
    if len(spreads) < 20:
        return np.nan
    arr = np.array(spreads)
    std = arr.std()
    if std == 0 or np.isnan(std):
        return np.nan
    return float(arr.mean() / std * np.sqrt(252))


def regime_split_ic(
    signal: pd.DataFrame,
    forward_ret: pd.DataFrame,
    regime_mask: pd.Series,
) -> dict:
    daily = daily_cross_sectional_ic(signal, forward_ret)
    m = regime_mask.reindex(daily.index).fillna(False)
    on = daily[m]
    off = daily[~m]
    return {
        "ic_regime_on": float(on.mean()) if len(on) else np.nan,
        "ic_regime_off": float(off.mean()) if len(off) else np.nan,
        "n_days_on": int(len(on)),
        "n_days_off": int(len(off)),
    }


def evaluate_feature_panel(
    signal: pd.DataFrame,
    close: pd.DataFrame,
    *,
    feature_name: str,
    hypothesis: str,
    neutralization: str,
    regime_mask: pd.Series | None = None,
    horizons: tuple[int, ...] = HORIZONS_EQUITY,
    full_metrics: bool = True,
) -> list[dict]:
    fwd = {h: forward_return_panel(close, h) for h in horizons}
    rows = []
    for h, fwd_h in fwd.items():
        row = {
            "feature": feature_name,
            "hypothesis": hypothesis,
            "neutralization": neutralization,
            "horizon": h,
            "ic_spearman": mean_ic(signal, fwd_h),
            "quintile_spread": pooled_quintile_spread(signal, fwd_h).get("spread_q5_q1", np.nan),
            "hit_ratio": pooled_hit_ratio(signal, fwd_h),
            "n_stock_days": int(stack_wide(signal).notna().sum()),
        }
        if full_metrics:
            boot = bootstrap_ic_stability(signal, fwd_h, n_draws=50)
            row.update(
                {
                    "bootstrap_survival": boot["survival_rate"],
                    "bootstrap_ic_p05": boot["p05"],
                    "bootstrap_ic_p95": boot["p95"],
                    "residual_sharpe_proxy": residual_sharpe_proxy(signal, fwd_h),
                }
            )
            ric = rolling_ic_series(signal, fwd_h)
            row["rolling_ic_last"] = float(ric.iloc[-1]) if len(ric) else np.nan
            if regime_mask is not None:
                row.update(regime_split_ic(signal, fwd_h, regime_mask))
        else:
            row.update(
                {
                    "bootstrap_survival": np.nan,
                    "bootstrap_ic_p05": np.nan,
                    "bootstrap_ic_p95": np.nan,
                    "residual_sharpe_proxy": np.nan,
                    "rolling_ic_last": np.nan,
                }
            )
        rows.append(row)
    return rows


def write_results(
    ic_rows: list[dict],
    neutral_rows: list[dict],
    regime_rows: list[dict],
    summary: dict,
    hypothesis_report: dict,
    *,
    results_dir: Path = Path("results"),
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ic_rows).to_csv(results_dir / "equity_signal_ic.csv", index=False)
    pd.DataFrame(neutral_rows).to_csv(results_dir / "equity_neutralization.csv", index=False)
    pd.DataFrame(regime_rows).to_csv(results_dir / "equity_regime_analysis.csv", index=False)
    with open(results_dir / "equity_signal_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    with open(results_dir / "equity_hypothesis_report.json", "w", encoding="utf-8") as f:
        json.dump(hypothesis_report, f, indent=2, default=str)
