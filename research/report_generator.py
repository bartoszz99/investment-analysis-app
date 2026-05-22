"""Institutional HTML research report."""

import html
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _fig_to_base64(path: str) -> str:
    import base64

    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("ascii")


def generate_institutional_report(
    df: pd.DataFrame,
    metrics: dict,
    equity: list[float],
    config_snapshot: dict,
    validation_summary: dict | None = None,
    lineage_path: str | None = None,
    output_path: str = "research/report.html",
) -> str:
    validation_summary = validation_summary or {}
    eq = pd.Series(equity, index=df.index[: len(equity)])
    dd = eq / eq.expanding().max() - 1
    rets = eq.pct_change().dropna()
    roll_sharpe = (rets.rolling(20).mean() / rets.rolling(20).std()) * (252**0.5)

    charts_dir = Path("research/charts")
    charts_dir.mkdir(parents=True, exist_ok=True)
    eq_path = charts_dir / "equity.png"
    plt.figure(figsize=(10, 4))
    eq.plot(title="Equity")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(eq_path, dpi=120)
    plt.close()

    dd_path = charts_dir / "drawdown.png"
    plt.figure(figsize=(10, 3))
    dd.plot(title="Drawdown", color="red")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(dd_path, dpi=120)
    plt.close()

    lineage_block = ""
    if lineage_path and Path(lineage_path).exists():
        lineage_block = f"<pre>{html.escape(Path(lineage_path).read_text(encoding='utf-8')[:3000])}</pre>"

    body = f"""
    <html><head><title>Research Report</title>
    <style>body{{font-family:Arial;margin:24px}} table{{border-collapse:collapse}}
    td,th{{border:1px solid #ccc;padding:6px}}</style></head><body>
    <h1>Quant Research Report</h1>
    <h2>Summary</h2>
    <table>
    <tr><th>Return %</th><td>{metrics.get('return_pct',0):.2f}</td></tr>
    <tr><th>Sharpe</th><td>{metrics.get('sharpe',0):.4f}</td></tr>
    <tr><th>Max DD %</th><td>{metrics.get('max_drawdown',0):.2f}</td></tr>
    <tr><th>Trades</th><td>{metrics.get('trades',0)}</td></tr>
    </table>
    <h2>Equity</h2><img src="data:image/png;base64,{_fig_to_base64(str(eq_path))}" width="800"/>
    <h2>Drawdown</h2><img src="data:image/png;base64,{_fig_to_base64(str(dd_path))}" width="800"/>
    <h2>Validation</h2><pre>{html.escape(json.dumps(validation_summary, indent=2, default=str))}</pre>
    <h2>Config</h2><pre>{html.escape(json.dumps(config_snapshot, indent=2))}</pre>
    <h2>Lineage</h2>{lineage_block}
  </body></html>
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(body, encoding="utf-8")
    print(f"Report: {output_path}")
    return output_path
