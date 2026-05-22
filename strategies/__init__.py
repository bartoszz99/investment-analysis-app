from strategies.base import BaseStrategy
from strategies.buy_and_hold import BuyAndHold
from strategies.sma_crossover import SmaCrossover

DEFAULT_STRATEGIES = [SmaCrossover(), BuyAndHold()]

__all__ = ["BaseStrategy", "BuyAndHold", "SmaCrossover", "DEFAULT_STRATEGIES"]
