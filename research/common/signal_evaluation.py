"""
Reusable signal evaluation — IC, quintiles, regime splits.
Pure hypothesis testing; no optimization.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from research.common.forward_returns import HORIZONS


def spearman_ic(signal: pd.Series, forward_ret: pd.Series) -> float:
    aligned = pd.concat([signal.rename("s"), forward_ret.rename("f")], axis=1).dropna()
    if len(aligned) < 10:
        return np.nan
    return float(aligned["s"].rank().corr(aligned["f"].rank()))


def pearson_corr(signal: pd.Series, forward_ret: pd.Series) -> float:
    aligned = pd.concat([signal, forward_ret], axis=1).dropna()
    if len(aligned) < 10:
        return np.nan
    c = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
    return float(c) if c == c else np.nan


def quintile_spread(signal: pd.Series, forward_ret: pd.Series, n: int = 5) -> dict:
    aligned = pd.concat([signal.rename("s"), forward_ret.rename("f")], axis=1).dropna()
    if len(aligned) < n * 5:
        return {"spread_q5_q1": np.nan, "quantile_means": {}}
    aligned["q"] = pd.qcut(aligned["s"].rank(method="first"), n, labels=False) + 1
    means = aligned.groupby("q")["f"].mean()
    return {
        "spread_q5_q1": float(means.get(n, np.nan) - means.get(1, np.nan)),
        "quantile_means": {int(k): float(v) for k, v in means.items()},
    }


def hit_ratio(signal: pd.Series, forward_ret: pd.Series) -> float:
    aligned = pd.concat([signal, forward_ret], axis=1).dropna()
    if len(aligned) < 5:
        return np.nan
    same_sign = (aligned.iloc[:, 0] * aligned.iloc[:, 1]) > 0
    return float(same_sign.mean())


def rolling_ic(signal: pd.Series, forward_ret: pd.Series, window: int = 60) -> pd.Series:
    aligned = pd.concat([signal, forward_ret], axis=1).dropna()
    if len(aligned) < window:
        return pd.Series(dtype=float)

    def _ic_slice(end: int) -> float:
        sl = aligned.iloc[end - window : end]
        return spearman_ic(sl.iloc[:, 0], sl.iloc[:, 1])

    vals = [_ic_slice(i) for i in range(window, len(aligned) + 1)]
    return pd.Series(vals, index=aligned.index[window - 1 : len(vals) + window - 1][: len(vals)])


def regime_ic(
    signal: pd.Series,
    forward_ret: pd.Series,
    regime_mask: pd.Series,
) -> dict:
    out = {}
    for label, mask in [("regime_on", regime_mask), ("regime_off", ~regime_mask)]:
        m = mask.reindex(signal.index).fillna(False)
        out[label] = spearman_ic(signal[m], forward_ret[m])
    return out


@dataclass
class SignalEvalResult:
    signal_name: str
    target: str
    horizon: int
    ic_spearman: float
    pearson: float
    quintile_spread: float
    hit_ratio: float
    ic_neutral: float | None = None
    n_obs: int = 0


def evaluate_signal(
    signal: pd.Series,
    forward_ret: pd.Series,
    *,
    signal_name: str = "signal",
    target: str = "target",
    horizon: int = 20,
    neutral_signal: pd.Series | None = None,
    regime_mask: pd.Series | None = None,
) -> dict:
    qs = quintile_spread(signal, forward_ret)
    result = {
        "signal": signal_name,
        "target": target,
        "horizon": horizon,
        "ic_spearman": spearman_ic(signal, forward_ret),
        "pearson": pearson_corr(signal, forward_ret),
        "quintile_spread": qs.get("spread_q5_q1", qs.get("spread", np.nan)),
        "quantile_means": qs["quantile_means"],
        "hit_ratio": hit_ratio(signal, forward_ret),
        "n_obs": int(pd.concat([signal, forward_ret], axis=1).dropna().shape[0]),
    }
    if neutral_signal is not None:
        result["ic_neutral"] = spearman_ic(neutral_signal, forward_ret)
    if regime_mask is not None:
        result["regime_ic"] = regime_ic(signal, forward_ret, regime_mask)
    return result


def evaluate_signal_horizons(
    signal: pd.Series,
    close: pd.Series,
    *,
    signal_name: str,
    target: str,
    horizons: tuple[int, ...] = HORIZONS,
    neutral_signal: pd.Series | None = None,
    regime_mask: pd.Series | None = None,
) -> list[dict]:
    from research.common.forward_returns import forward_return

    rows = []
    for h in horizons:
        fwd = forward_return(close, h)
        rows.append(
            evaluate_signal(
                signal,
                fwd,
                signal_name=signal_name,
                target=target,
                horizon=h,
                neutral_signal=neutral_signal,
                regime_mask=regime_mask,
            )
        )
    return rows


def long_short_decile_curve(signal: pd.Series, forward_ret: pd.Series, n: int = 10) -> pd.Series:
    """Mean forward return per decile (for plotting)."""
    aligned = pd.concat([signal.rename("s"), forward_ret.rename("f")], axis=1).dropna()
    if len(aligned) < n * 3:
        return pd.Series(dtype=float)
    aligned["d"] = pd.qcut(aligned["s"].rank(method="first"), n, labels=False) + 1
    return aligned.groupby("d")["f"].mean()
