"""
Cross-sectional IC, quintiles, neutralized metrics per feature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.common.signal_evaluation import hit_ratio, quintile_spread
from research.equity_alpha_lab.feature_neutralization import neutralize_all
from research.equity_alpha_lab.forward_returns import apply_cost_haircut, build_forward_returns_wide
from research.equity_alpha_lab.universe import EquityPanel, get_sector


def _stack(wide: pd.DataFrame) -> pd.Series:
    return wide.stack(future_stack=True)


def daily_ic(signal: pd.DataFrame, fwd: pd.DataFrame) -> pd.Series:
    cols = signal.columns.intersection(fwd.columns)
    s, f = signal[cols], fwd[cols]
    ok = (s.notna() & f.notna()).sum(axis=1) >= 15
    return s.rank(axis=1).corrwith(f.rank(axis=1), axis=1).loc[ok].dropna()


def mean_ic(signal: pd.DataFrame, fwd: pd.DataFrame) -> float:
    d = daily_ic(signal, fwd)
    return float(d.mean()) if len(d) else np.nan


def bootstrap_survival(signal: pd.DataFrame, fwd: pd.DataFrame, n: int = 50) -> float:
    d = daily_ic(signal, fwd).dropna()
    if len(d) < 30:
        return np.nan
    rng = np.random.default_rng(42)
    means = [float(d.iloc[rng.integers(0, len(d), len(d))].mean()) for _ in range(n)]
    return float((np.array(means) > 0).mean()) if d.mean() > 0 else float((np.array(means) < 0).mean())


def tail_spread(signal: pd.DataFrame, fwd: pd.DataFrame, tail_pct: float = 0.2) -> dict:
    """Q5-Q1 spread on top/bottom signal days (tail asymmetry focus)."""
    daily = signal.rank(axis=1).mean(axis=1)
    q_hi = daily.quantile(1 - tail_pct)
    q_lo = daily.quantile(tail_pct)
    hi_days = daily >= q_hi
    lo_days = daily <= q_lo
    s_hi, f_hi = _stack(signal.loc[hi_days]), _stack(fwd.loc[hi_days])
    s_lo, f_lo = _stack(signal.loc[lo_days]), _stack(fwd.loc[lo_days])
    qs_hi = quintile_spread(s_hi, f_hi) if len(s_hi) > 50 else {"spread_q5_q1": np.nan}
    qs_lo = quintile_spread(s_lo, f_lo) if len(s_lo) > 50 else {"spread_q5_q1": np.nan}
    return {
        "spread_tail_high_signal_days": qs_hi.get("spread_q5_q1", np.nan),
        "spread_tail_low_signal_days": qs_lo.get("spread_q5_q1", np.nan),
    }


def oos_ic_split(signal: pd.DataFrame, fwd: pd.DataFrame, train_frac: float = 0.8) -> dict:
    d = daily_ic(signal, fwd)
    if len(d) < 40:
        return {"ic_in_sample": np.nan, "ic_out_of_sample": np.nan}
    cut = int(len(d) * train_frac)
    return {
        "ic_in_sample": float(d.iloc[:cut].mean()),
        "ic_out_of_sample": float(d.iloc[cut:].mean()),
    }


def evaluate_market_breadth(
    signal_wide: pd.DataFrame,
    close: pd.DataFrame,
    fwd_dict: dict[int, pd.DataFrame],
    *,
    feature: str,
    category: str,
) -> tuple[list[dict], list[dict]]:
    from research.common.signal_evaluation import spearman_ic

    sig = signal_wide.iloc[:, 0]
    ic_rows = []
    for horizon, fwd_w in fwd_dict.items():
        fwd_ew = fwd_w.mean(axis=1)
        aligned = pd.concat([sig, fwd_ew], axis=1).dropna()
        ic = spearman_ic(aligned.iloc[:, 0], aligned.iloc[:, 1]) if len(aligned) > 30 else np.nan
        qs = quintile_spread(aligned.iloc[:, 0], aligned.iloc[:, 1])
        spread = qs.get("spread_q5_q1", np.nan)
        ic_rows.append(
            {
                "feature": feature,
                "category": category,
                "neutralization": "market_ts",
                "horizon": horizon,
                "ic_spearman": ic,
                "quintile_spread": spread,
                "quintile_spread_after_cost": apply_cost_haircut(spread, horizon),
                "hit_ratio": hit_ratio(aligned.iloc[:, 0], aligned.iloc[:, 1]),
                "leakage_risk": "LOW",
                "note": "market-level time-series IC vs EW forward return",
            }
        )
    ic5 = next((r["ic_spearman"] for r in ic_rows if r["horizon"] == 5), np.nan)
    neu = {
        "feature": feature,
        "category": category,
        "ic_raw_5d": ic5,
        "ic_sector_5d": np.nan,
        "ic_beta_5d": np.nan,
        "ic_momentum_5d": np.nan,
        "leakage_risk": "LOW",
    }
    return ic_rows, [neu]


def evaluate_feature(
    signal: pd.DataFrame,
    close: pd.DataFrame,
    fwd_dict: dict[int, pd.DataFrame],
    *,
    feature: str,
    category: str,
    sectors: dict[str, str],
    spy_ret: pd.Series,
    regime_mask: pd.Series,
    leakage_risk: str = "LOW",
) -> tuple[list[dict], list[dict]]:
    ic_rows: list[dict] = []
    neutral = neutralize_all(signal, sectors=sectors, spy_ret=spy_ret, close=close)
    ic5 = {k: mean_ic(v, fwd_dict[5]) for k, v in neutral.items()}
    neu_row = {
        "feature": feature,
        "category": category,
        "ic_raw_5d": ic5.get("raw", np.nan),
        "ic_sector_5d": ic5.get("sector", np.nan),
        "ic_beta_5d": ic5.get("beta", np.nan),
        "ic_momentum_5d": ic5.get("momentum", np.nan),
        "leakage_risk": leakage_risk,
    }

    for horizon, fwd in fwd_dict.items():
        for neu_name, neu_sig in neutral.items():
            ic = mean_ic(neu_sig, fwd)
            qs = quintile_spread(_stack(neu_sig), _stack(fwd))
            spread = qs.get("spread_q5_q1", np.nan)
            row = {
                "feature": feature,
                "category": category,
                "neutralization": neu_name,
                "horizon": horizon,
                "ic_spearman": ic,
                "quintile_spread": spread,
                "quintile_spread_after_cost": apply_cost_haircut(spread, horizon),
                "hit_ratio": hit_ratio(_stack(neu_sig), _stack(fwd)),
                "leakage_risk": leakage_risk,
            }
            if neu_name == "raw":
                d_ic = daily_ic(neu_sig, fwd)
                row["rolling_ic_last"] = float(
                    d_ic.rolling(60, min_periods=20).mean().iloc[-1]
                ) if len(d_ic) >= 20 else np.nan
                row["bootstrap_survival"] = bootstrap_survival(neu_sig, fwd)
                row.update(oos_ic_split(neu_sig, fwd))
                row.update(tail_spread(neu_sig, fwd))
                m = regime_mask.reindex(d_ic.index).fillna(False)
                row["ic_regime_on"] = float(d_ic[m.reindex(d_ic.index, fill_value=False)].mean()) if m.any() else np.nan
                row["ic_regime_off"] = float(d_ic[~m.reindex(d_ic.index, fill_value=False)].mean()) if (~m).any() else np.nan
            ic_rows.append(row)

    return ic_rows, [neu_row]


def run_all_feature_tests(
    features: dict[str, pd.DataFrame],
    panel: EquityPanel,
    *,
    meta: dict,
) -> tuple[list[dict], list[dict]]:
    import yfinance as yf

    close = panel.close
    sectors = {t: get_sector(t) for t in close.columns}
    spy = yf.Ticker("SPY").history(period="3y", auto_adjust=True)
    if spy.index.tz:
        spy.index = spy.index.tz_localize(None)
    spy_close = spy["Close"].reindex(panel.calendar).ffill()
    spy_ret = spy_close.pct_change()
    regime_mask = spy_close.shift(1) > spy_close.shift(1).rolling(200, min_periods=200).mean()

    fwd = build_forward_returns_wide(close)
    ic_all, neu_all = [], []

    for name, sig in features.items():
        if name.startswith("earn_"):
            risk = "HIGH_LEAKAGE_RISK"
            cat = "A_post_earnings"
            rows, neu = evaluate_feature(
                sig, close, fwd,
                feature=name, category=cat, sectors=sectors,
                spy_ret=spy_ret, regime_mask=regime_mask, leakage_risk=risk,
            )
        elif name.startswith("breadth_"):
            rows, neu = evaluate_market_breadth(
                sig, close, fwd, feature=name, category="B_breadth",
            )
        else:
            rows, neu = evaluate_feature(
                sig, close, fwd,
                feature=name, category="C_liquidity", sectors=sectors,
                spy_ret=spy_ret, regime_mask=regime_mask, leakage_risk="LOW",
            )
        ic_all.extend(rows)
        neu_all.extend(neu)

    return ic_all, neu_all
