"""
Combine axis scores into hypothesis-level classification.
NOT alpha — investability of the idea as a coherent story.
"""

from __future__ import annotations

from enum import Enum

import numpy as np

WEIGHT_FUNDAMENTAL = 0.4
WEIGHT_TECHNICAL = 0.3
WEIGHT_STRUCTURAL = 0.3


class IdeaVerdict(str, Enum):
    INVESTABLE_HYPOTHESIS = "investable_hypothesis"  # >= 0.5
    WEAK_MONITOR = "weak_monitor"  # 0.2 - 0.5
    NOISE = "noise"  # < 0.2


def combine_scores(
    score_fundamental: float,
    score_technical: float,
    score_structural: float,
    *,
    structural_class: str,
    leakage_penalty: bool = False,
) -> dict:
    sf = score_fundamental if score_fundamental == score_fundamental else 0.0
    st = score_technical if score_technical == score_technical else 0.0
    ss = score_structural if score_structural == score_structural else 0.0

    # Structural override: exposure/noise caps composite
    if structural_class == "MARKET_EXPOSURE":
        ss = min(ss, 0.25)
    elif structural_class == "NOISE":
        ss = 0.0

    raw = WEIGHT_FUNDAMENTAL * sf + WEIGHT_TECHNICAL * st + WEIGHT_STRUCTURAL * ss
    if leakage_penalty:
        raw *= 0.5

    # Composite in [0,1] for classification (magnitude of conviction, not direction)
    idea_score = float(np.clip(abs(raw), 0.0, 1.0))
    directional_score = float(np.clip(raw, -1.0, 1.0))

    if idea_score >= 0.5:
        verdict = IdeaVerdict.INVESTABLE_HYPOTHESIS.value
        label = "Coherent investable hypothesis (not confirmed alpha)"
    elif idea_score >= 0.2:
        verdict = IdeaVerdict.WEAK_MONITOR.value
        label = "Weak idea — monitor only"
    else:
        verdict = IdeaVerdict.NOISE.value
        label = "Likely noise — discard"

    return {
        "idea_score": idea_score,
        "directional_score": directional_score,
        "verdict": verdict,
        "verdict_label": label,
        "weights": {
            "fundamental": WEIGHT_FUNDAMENTAL,
            "technical": WEIGHT_TECHNICAL,
            "structural": WEIGHT_STRUCTURAL,
        },
        "components": {
            "fundamental": sf,
            "technical": st,
            "structural": ss,
        },
    }
