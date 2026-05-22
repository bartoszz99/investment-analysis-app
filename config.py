TICKER = "AAPL"
PERIOD = "1mo"
SMA_SHORT = 7
SMA_LONG = 30
CHART_FILE = "chart.png"
SHOW_PLOT = True

# --- Multi-asset ETF universe (US) ---
ETF_UNIVERSE = ("SPY", "QQQ", "IWM", "XLK", "XLF", "XLE")
MULTI_ASSET_PERIOD = "1y"
MULTI_ASSET_START_CAPITAL = 100_000.0
TRADE_FEE_BPS = 2.0  # configurable 1–5 bps per trade
SLIPPAGE_VOL_COEF = 0.10  # slippage scales with trailing vol
MAX_WEIGHT_PER_ASSET = 0.4
MOMENTUM_TOP_K = 2
SHORT_ENABLED = False
PORTFOLIO_METHOD = "equal_weight"  # "equal_weight" | "risk_parity"
REBALANCE_FREQ = "daily"
