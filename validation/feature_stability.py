"""Rolling feature importance stability."""

import numpy as np
import pandas as pd


def importance_stability(
    importance_by_window: list[dict[str, float]],
) -> dict:
    """
    importance_by_window: list of feature->importance dicts per WF window.
    """
    if not importance_by_window:
        return {"stability_score": 0.0, "unstable_features": []}
    features = set()
    for d in importance_by_window:
        features.update(d.keys())
    unstable = []
    scores = []
    for f in features:
        vals = [d.get(f, 0.0) for d in importance_by_window]
        m = np.mean(vals)
        s = np.std(vals)
        cv = s / (abs(m) + 1e-8)
        scores.append(cv)
        if cv > 1.0:
            unstable.append(f)
    stability = 1.0 / (1.0 + np.mean(scores)) if scores else 0.0
    return {
        "stability_score": float(stability),
        "unstable_features": unstable,
        "warning": "WARNING: unstable feature importance" if unstable else "OK",
    }
