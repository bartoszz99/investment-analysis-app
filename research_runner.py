"""
Market structure research entry point.
Runs breadth + liquidity hypothesis labs (isolated from production).
"""

from research.breadth.breadth_runner import run_breadth_research
from research.liquidity.liquidity_runner import run_liquidity_research
from research.passive_flow.passive_flow_runner import run_passive_flow_research


def run_market_structure_research(period: str = "2y") -> dict:
    print("=" * 60)
    print("MARKET STRUCTURE RESEARCH LAB")
    print("Hypothesis-driven — no portfolio optimization")
    print("=" * 60)
    breadth = run_breadth_research(period)
    liquidity = run_liquidity_research(period)
    passive = run_passive_flow_research(period="10y")
    return {"breadth": breadth, "liquidity": liquidity, "passive_flow": passive}


if __name__ == "__main__":
    run_market_structure_research()
