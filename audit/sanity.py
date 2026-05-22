"""Institutional sanity checks — OK / WARNING / FAIL."""

from dataclasses import dataclass

import pandas as pd

from layers.backtest_engine import run_backtest
from layers.execution_layer import POSITION_SIZE
from strategies import BaseStrategy, BuyAndHold, SmaCrossover


@dataclass
class SanityResult:
    status: str
    check: str
    reason: str


def run_institutional_sanity(
    df: pd.DataFrame,
    results: dict[str, dict],
    strategies: list[BaseStrategy] | None = None,
) -> list[SanityResult]:
    strategies = strategies or []
    out: list[SanityResult] = []
    close = df["Close"]

    bh_name = "Buy & Hold"
    if bh_name in results:
        m = results[bh_name]
        passive = (close.iloc[-1] / close.iloc[0] - 1) * 100 * POSITION_SIZE
        if m["trades"] < 1:
            out.append(SanityResult("FAIL", "buy_hold_trades", "No trades after execution lag"))
        elif abs(m["return_pct"] - passive) > 10:
            out.append(
                SanityResult(
                    "WARNING",
                    "buy_hold_bounds",
                    f"Return {m['return_pct']:.1f}% vs passive ~{passive:.1f}%",
                )
            )
        else:
            out.append(SanityResult("OK", "buy_hold_bounds", "Consistent with passive benchmark"))

    for name, m in results.items():
        if m["sharpe"] > 3.0 and m["return_pct"] > 50:
            out.append(
                SanityResult(
                    "WARNING",
                    f"sharpe_spike_{name}",
                    f"Sharpe={m['sharpe']:.2f} return={m['return_pct']:.1f}% — verify leakage",
                )
            )
        days = len(df)
        if days > 0 and m["trades"] > days * 0.5:
            out.append(
                SanityResult(
                    "WARNING",
                    f"turnover_{name}",
                    f"Trades={m['trades']} on {days} bars — high turnover",
                )
            )

    sma = next((s for s in strategies if isinstance(s, SmaCrossover)), None)
    if sma and sma.name in results and results[sma.name]["trades"] == 0:
        if len(df) < sma.sma_long + 5:
            out.append(SanityResult("WARNING", "sma_warmup", "Zero trades — insufficient warmup"))
        else:
            out.append(SanityResult("FAIL", "sma_zero_trades", "Zero trades with enough data"))

    return out


def print_sanity_results(results: list[SanityResult]) -> bool:
    ok = True
    print("\n=== INSTITUTIONAL SANITY ===")
    for r in results:
        print(f"[{r.status}] {r.check}: {r.reason}")
        if r.status == "FAIL":
            ok = False
    if ok and not any(x.status == "WARNING" for x in results):
        print("[SANITY OK] Research checks passed")
    return ok
