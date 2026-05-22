"""Data layer facade — immutable OHLCV only."""
from layers.data_layer import (
    MultiAssetOHLCV,
    align_and_clean,
    fetch_raw,
    load_market_data,
    load_multi_asset,
    synchronize_panels,
    validate_market_data,
)


def fetch_data(ticker: str, period: str = "1mo"):
    return fetch_raw(ticker, period)


def clean_data(df, ticker: str | None = None):
    return align_and_clean(df, ticker)
