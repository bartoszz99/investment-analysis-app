"""
Alternative signal research tests — IC, forward correlation, quintiles, regime splits.
Post-hoc hypothesis testing only; does not modify production stack.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def forward_returns(close: pd.Series, horizons: tuple[int, ...] = (1, 5, 20)) -> pd.DataFrame:
    """Forward cumulative returns from t+1 through t+h (close-to-close)."""
    ret = close.pct_change()
    out = {}
    for h in horizons:
        # close[t+h]/close[t] - 1 approximates sum of daily returns t+1..t+h
        out[f"fwd_{h}d"] = close.shift(-h) / close - 1.0
    return pd.DataFrame(out, index=close.index)


def information_coefficient(signal: pd.Series, forward_ret: pd.Series) -> float:
    """Spearman rank IC between signal[t] and forward return."""
    aligned = pd.concat([signal.rename("s"), forward_ret.rename("f")], axis=1).dropna()
    if len(aligned) < 10:
        return np.nan
    return float(aligned["s"].rank().corr(aligned["f"].rank()))


def ic_series_rolling(signal: pd.Series, forward_ret: pd.Series, window: int = 60) -> pd.Series:
    aligned = pd.concat([signal, forward_ret], axis=1).dropna()
    if len(aligned) < window:
        return pd.Series(dtype=float)

    def _ic(row_idx: int) -> float:
        sl = aligned.iloc[row_idx - window : row_idx]
        if len(sl) < 10:
            return np.nan
        return sl.iloc[:, 0].rank().corr(sl.iloc[:, 1].rank())

    vals = [_ic(i) for i in range(window, len(aligned) + 1)]
    idx = aligned.index[window - 1 :]
    return pd.Series(vals, index=idx[: len(vals)])


def quintile_spread(signal: pd.Series, forward_ret: pd.Series, n_quantiles: int = 5) -> dict:
    """Long Q5 - Q1 forward return spread (diagnostic)."""
    aligned = pd.concat([signal.rename("s"), forward_ret.rename("f")], axis=1).dropna()
    if len(aligned) < n_quantiles * 5:
        return {"spread": np.nan, "n_obs": len(aligned)}
    aligned["q"] = pd.qcut(aligned["s"].rank(method="first"), n_quantiles, labels=False) + 1
    means = aligned.groupby("q")["f"].mean()
    return {
        "spread_q5_q1": float(means.get(n_quantiles, np.nan) - means.get(1, np.nan)),
        "quantile_means": {int(k): float(v) for k, v in means.items()},
        "n_obs": len(aligned),
    }


def normalize_series_index(series: pd.Series, reference_index: pd.DatetimeIndex) -> pd.Series:
    """Align signal index to reference calendar (date-normalized, tz stripped)."""
    from research.alternative_data.base import normalize_index

    ref = normalize_index(reference_index)
    s = series.copy()
    s.index = normalize_index(s.index)
    return s.reindex(ref)


def neutralize_signal_vs_spy(
    signal: pd.Series,
    spy_returns: pd.Series,
) -> pd.Series:
    """Residual signal after OLS on SPY returns (diagnostic IC test)."""
    signal = normalize_series_index(signal, spy_returns.index)
    aligned = pd.concat([signal.rename("s"), spy_returns.rename("r")], axis=1).dropna()
    if len(aligned) < 10:
        return pd.Series(dtype=float)
    y = aligned["s"].to_numpy()
    x = np.column_stack([np.ones(len(y)), aligned["r"].to_numpy()])
    coeffs, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ coeffs
    return pd.Series(resid, index=aligned.index, name="signal_neutral")


def regime_mask_spy_vol(spy_close: pd.Series, high_quantile: float = 0.70, window: int = 20) -> pd.Series:
    lagged = spy_close.shift(1)
    vol = lagged.pct_change().rolling(window, min_periods=window).std()
    thresh = vol.expanding(min_periods=window).quantile(high_quantile)
    return (vol >= thresh).fillna(False)


def test_signal_vs_target(
    signal,
    target_close: pd.Series,
    spy_close: pd.Series,
    regime_mask: pd.Series,
    horizons: tuple[int, ...] = (1, 5, 20),
) -> dict:
    s = normalize_series_index(signal.series, target_close.index)
    spy_ret = spy_close.pct_change()
    s_neutral = neutralize_signal_vs_spy(s, spy_ret)
    fwd = forward_returns(target_close, horizons)

    result = {
        "signal": signal.metadata.name,
        "leakage_risk": signal.metadata.leakage_risk.value,
        "source": signal.metadata.source,
        "horizons": {},
        "ic_neutral_vs_spy": {},
        "quintile_fwd_20d": {},
        "regime_ic_fwd_20d": {},
    }

    h20 = fwd["fwd_20d"]
    for h in horizons:
        col = f"fwd_{h}d"
        fr = fwd[col]
        result["horizons"][col] = {
            "ic": information_coefficient(s, fr),
            "pearson": float(s.corr(fr)) if s.corr(fr) == s.corr(fr) else np.nan,
        }
        result["ic_neutral_vs_spy"][col] = information_coefficient(s_neutral, fr)

    result["quintile_fwd_20d"] = quintile_spread(s, h20)
    result["quintile_fwd_20d_neutral"] = quintile_spread(s_neutral, h20)

    for label, mask in [("high_vol", regime_mask), ("low_vol", ~regime_mask)]:
        m = mask.reindex(s.index).fillna(False)
        sub_s = s[m]
        sub_f = h20[m]
        result["regime_ic_fwd_20d"][label] = information_coefficient(sub_s, sub_f)

    result["ic_mean_rolling_60d"] = float(
        ic_series_rolling(s, h20, 60).mean()
    ) if len(s.dropna()) > 60 else np.nan

    return result


def run_all_signal_tests(
    registry,
    close_wide: pd.DataFrame,
    benchmark_ticker: str = "SPY",
) -> tuple[list[dict], pd.DataFrame, pd.DataFrame]:
    """Test each signal vs each ETF forward returns + SPY neutralization."""
    from research.alternative_data.base import normalize_index

    close_wide = close_wide.copy()
    close_wide.index = normalize_index(close_wide.index)
    spy = close_wide[benchmark_ticker]
    regime = regime_mask_spy_vol(spy)
    all_results: list[dict] = []
    ic_rows: list[dict] = []
    regime_rows: list[dict] = []

    for sig in registry:
        for ticker in close_wide.columns:
            tests = test_signal_vs_target(sig, close_wide[ticker], spy, regime)
            tests["target_etf"] = ticker
            all_results.append(tests)

            ic_rows.append(
                {
                    "signal": sig.metadata.name,
                    "target": ticker,
                    "ic_1d": tests["horizons"].get("fwd_1d", {}).get("ic"),
                    "ic_5d": tests["horizons"].get("fwd_5d", {}).get("ic"),
                    "ic_20d": tests["horizons"].get("fwd_20d", {}).get("ic"),
                    "ic_20d_neutral_spy": tests["ic_neutral_vs_spy"].get("fwd_20d"),
                    "ic_rolling_60d_mean": tests.get("ic_mean_rolling_60d"),
                    "quintile_spread_20d": tests["quintile_fwd_20d"].get("spread_q5_q1"),
                    "quintile_spread_20d_neutral": tests["quintile_fwd_20d_neutral"].get("spread_q5_q1"),
                    "leakage_risk": sig.metadata.leakage_risk.value,
                }
            )
            regime_rows.append(
                {
                    "signal": sig.metadata.name,
                    "target": ticker,
                    "ic_high_vol": tests["regime_ic_fwd_20d"].get("high_vol"),
                    "ic_low_vol": tests["regime_ic_fwd_20d"].get("low_vol"),
                }
            )

    ic_df = pd.DataFrame(ic_rows)
    regime_df = pd.DataFrame(regime_rows)
    return all_results, ic_df, regime_df


def rank_signals_by_neutral_ic(ic_df: pd.DataFrame, min_obs_threshold: float = 0.0) -> pd.DataFrame:
    """Rank signals with stable neutral IC (primary research goal)."""
    agg = (
        ic_df.groupby("signal")
        .agg(
            mean_ic_20d=("ic_20d", "mean"),
            mean_ic_20d_neutral=("ic_20d_neutral_spy", "mean"),
            std_ic_neutral=("ic_20d_neutral_spy", "std"),
            n_targets=("target", "count"),
        )
        .reset_index()
    )
    agg["stable_neutral_ic"] = (
        agg["mean_ic_20d_neutral"].abs() > 0.03
    ) & (agg["std_ic_neutral"] < 0.15)
    return agg.sort_values("mean_ic_20d_neutral", key=abs, ascending=False)


def save_research_outputs(
    all_results: list[dict],
    ic_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    registry_summary: list[dict],
    results_dir: str | Path = "results",
) -> None:
    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)

    ranked = rank_signals_by_neutral_ic(ic_df)
    payload = {
        "registry": registry_summary,
        "n_tests": len(all_results),
        "top_neutral_ic_signals": ranked.head(10).to_dict(orient="records"),
        "tests_sample": all_results[:20],
        "research_goal": "Find small but stable IC after SPY neutralization",
    }
    with open(out / "alternative_signal_tests.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)

    ic_df.to_csv(out / "ic_analysis.csv", index=False)
    regime_df.to_csv(out / "regime_signal_analysis.csv", index=False)
