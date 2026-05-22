"""Leakage audit v2 — lineage, feature scores, suspicious metrics."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from layers.feature_store import FeatureStore


@dataclass
class AuditFinding:
    severity: str
    component: str
    message: str


@dataclass
class LeakageReport:
    findings: list[AuditFinding] = field(default_factory=list)
    leakage_score: float = 0.0
    feature_scores: dict[str, float] = field(default_factory=dict)

    def add(self, severity: str, component: str, message: str) -> None:
        self.findings.append(AuditFinding(severity, component, message))

    def finalize(self) -> "LeakageReport":
        weights = {"FAIL": 1.0, "WARNING": 0.4, "OK": 0.0}
        if not self.findings:
            self.leakage_score = 0.0
            return self
        total = sum(weights.get(f.severity, 0.2) for f in self.findings)
        feat_penalty = np.mean(list(self.feature_scores.values())) if self.feature_scores else 0
        self.leakage_score = min(1.0, total / len(self.findings) + feat_penalty * 0.2)
        return self


def feature_lineage(store: FeatureStore | None = None) -> dict[str, dict]:
    store = store or FeatureStore()
    lineage = {}
    for name, spec in store._registry.items():
        lineage[name] = {
            "category": spec.category,
            "lookback": spec.lookback,
            "lag": spec.lag,
            "dependencies": list(spec.dependencies),
            "leakage_safe": spec.leakage_safe,
        }
    return lineage


def score_features(df: pd.DataFrame, store: FeatureStore | None = None) -> dict[str, float]:
    store = store or FeatureStore()
    built = store.build(df)
    scores = {}
    close = df["Close"]
    for name in store.list_features():
        if name not in built.columns:
            continue
        s = built[name]
        future_ret = close.pct_change().shift(-1)
        corr = abs(s.corr(future_ret)) if s.notna().sum() > 10 else 0
        scores[name] = min(1.0, corr) if not np.isnan(corr) else 0.0
    return scores


def suspicious_correlation_detector(df: pd.DataFrame, threshold: float = 0.95) -> list[str]:
    store = FeatureStore()
    built = store.build(df)
    future = df["Close"].pct_change().shift(-1)
    flagged = []
    for col in built.columns:
        if col in df.columns:
            continue
        c = built[col].corr(future)
        if c is not None and abs(c) >= threshold:
            flagged.append(col)
    return flagged


def flag_metric_anomalies(metrics: dict, trades: int, days: int) -> list[AuditFinding]:
    findings = []
    sharpe = metrics.get("sharpe", 0)
    ret = metrics.get("return_pct", 0)
    if sharpe > 3.0:
        findings.append(AuditFinding("WARNING", "metrics", f"Unusually high Sharpe={sharpe:.2f}"))
    if days > 0 and trades > days * 0.4:
        findings.append(AuditFinding("WARNING", "metrics", f"Unrealistic turnover trades={trades}/{days}"))
    if ret > 100:
        findings.append(AuditFinding("WARNING", "metrics", f"Return {ret:.1f}% — verify execution model"))
    return findings


def run_leakage_audit(df: pd.DataFrame, context: str = "dataframe") -> LeakageReport:
    report = LeakageReport()
    store = FeatureStore()
    report.feature_scores = score_features(df, store)

    for name, score in report.feature_scores.items():
        if score > 0.9:
            report.add("WARNING", name, f"High corr with future return ({score:.2f})")

    flagged = suspicious_correlation_detector(df)
    for f in flagged:
        report.add("FAIL", f, "Suspicious correlation with next-bar return")

    if "position" in df.columns:
        probe = df.head(min(20, len(df))).copy()
        if len(probe) >= 4:
            probe["position"] = 0
            probe.iloc[2, probe.columns.get_loc("position")] = 1
            from layers.backtest_engine import run_backtest

            eq = run_backtest(probe)["equity"]
            if len(eq) >= 3 and eq[2] != eq[1]:
                report.add("FAIL", context, "Possible same-bar execution")
            else:
                report.add("OK", context, "Execution lag OK (Open t+1)")

    report.add("OK", "feature_store", f"Registered {len(store.list_features())} causal features")
    return report.finalize()


def audit_strategy_pipeline(base_df: pd.DataFrame, strategy) -> LeakageReport:
    report = LeakageReport()
    try:
        sig = strategy.apply(base_df.copy())
        report = run_leakage_audit(sig, strategy.name)
        if "sharpe" in dir(strategy):
            pass
    except Exception as exc:
        report.add("FAIL", strategy.name, str(exc))
    return report.finalize()


def strategy_leakage_score(base_df: pd.DataFrame, strategies: list) -> dict[str, float]:
    return {s.name: audit_strategy_pipeline(base_df, s).leakage_score for s in strategies}


def generate_audit_report(
    base_df: pd.DataFrame,
    strategies: list,
    results: dict | None = None,
    output_path: str = "leakage_audit_report.txt",
) -> LeakageReport:
    lines = ["=== LEAKAGE AUDIT REPORT v2 ===", ""]
    combined = LeakageReport()
    store = FeatureStore()

    lines.append("--- Feature lineage ---")
    for name, meta in feature_lineage(store).items():
        lines.append(f"  {name}: {meta}")
    lines.append("")

    lines.append("--- Dependency graph ---")
    for parent, deps in store.dependency_graph().items():
        lines.append(f"  {parent} <- {deps}")
    lines.append("")

    feat_scores = score_features(base_df, store)
    lines.append("--- Feature leakage scores ---")
    for n, sc in feat_scores.items():
        lines.append(f"  {n}: {sc:.3f}")
    combined.feature_scores = feat_scores
    lines.append("")

    for strategy in strategies:
        rep = audit_strategy_pipeline(base_df, strategy)
        lines.append(f"--- {strategy.name} (score={rep.leakage_score:.2f}) ---")
        for f in rep.findings:
            lines.append(f"  [{f.severity}] {f.component}: {f.message}")
        if results and strategy.name in results:
            for af in flag_metric_anomalies(results[strategy.name], results[strategy.name].get("trades", 0), len(base_df)):
                lines.append(f"  [{af.severity}] {af.component}: {af.message}")
        combined.findings.extend(rep.findings)
        lines.append("")

    combined.finalize()
    text = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"Audit report saved: {output_path}")
    return combined
