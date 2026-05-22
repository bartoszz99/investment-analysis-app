"""
Sentiment proxies — NO LLM, NO scraping engine.
Extensible placeholder interfaces for future data vendors.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import yfinance as yf

from research.alternative_data.base import (
    AlternativeSignal,
    LeakageRisk,
    SignalMetadata,
    align_to_calendar,
    apply_lag,
    expanding_zscore,
)


class SentimentSource(ABC):
    """Placeholder interface for external sentiment feeds."""

    @abstractmethod
    def fetch(self, calendar: pd.DatetimeIndex, period: str) -> pd.Series:
        ...


class NewsSentimentPlaceholder(SentimentSource):
    """
    Vendor hook — returns NaN until wired to a licensed news sentiment API.
    LEAKAGE_RISK=HIGH if used without verified publication timestamps.
    """

    def fetch(self, calendar: pd.DatetimeIndex, period: str) -> pd.Series:
        return pd.Series(np.nan, index=calendar, name="news_sentiment_placeholder")


class SocialSentimentPlaceholder(SentimentSource):
    """Reddit/Twitter hook — no scraping; NaN by default."""

    def fetch(self, calendar: pd.DatetimeIndex, period: str) -> pd.Series:
        return pd.Series(np.nan, index=calendar, name="social_sentiment_placeholder")


def fear_greed_vix_proxy(calendar: pd.DatetimeIndex, period: str, lag_days: int = 1) -> pd.Series:
    """
    CNN Fear & Greed substitute: inverted expanding z-score of VIX.
    Lower VIX -> higher 'greed' score.
    """
    vix = yf.Ticker("^VIX").history(period=period)
    if vix.empty:
        return pd.Series(np.nan, index=calendar)
    close = vix["Close"].copy()
    if close.index.tz:
        close.index = close.index.tz_localize(None)
    close.index = pd.DatetimeIndex([pd.Timestamp(d).normalize() for d in close.index])
    aligned = align_to_calendar(close, calendar)
    z = expanding_zscore(apply_lag(aligned, lag_days))
    return -z  # invert: high VIX = fear


def build_sentiment_signals(
    calendar: pd.DatetimeIndex,
    period: str = "1y",
    lag_days: int = 1,
) -> dict[str, AlternativeSignal]:
    fg = fear_greed_vix_proxy(calendar, period, lag_days)
    news = NewsSentimentPlaceholder().fetch(calendar, period)
    social = SocialSentimentPlaceholder().fetch(calendar, period)

    signals = {
        "sentiment_fear_greed_proxy": AlternativeSignal(
            metadata=SignalMetadata(
                name="sentiment_fear_greed_proxy",
                source="^VIX inverted z-score (Fear & Greed proxy)",
                lag_days=lag_days,
                update_frequency="daily",
                leakage_risk=LeakageRisk.LOW,
                timestamp_assumption="VIX close t-1 -> signal t",
                lag_policy=f"shift({lag_days}) on VIX close",
                description="VIX-based fear/greed proxy (no CNN API)",
            ),
            series=fg,
        ),
        "sentiment_news_placeholder": AlternativeSignal(
            metadata=SignalMetadata(
                name="sentiment_news_placeholder",
                source="NewsSentimentPlaceholder (unwired)",
                lag_days=lag_days,
                update_frequency="daily",
                leakage_risk=LeakageRisk.HIGH,
                timestamp_assumption="N/A until vendor wired",
                lag_policy="TBD per vendor SLA",
                description="Extensible news sentiment hook",
            ),
            series=news,
        ),
        "sentiment_social_placeholder": AlternativeSignal(
            metadata=SignalMetadata(
                name="sentiment_social_placeholder",
                source="SocialSentimentPlaceholder (unwired)",
                lag_days=lag_days,
                update_frequency="daily",
                leakage_risk=LeakageRisk.HIGH,
                timestamp_assumption="N/A — no scraping engine",
                lag_policy="TBD",
                description="Extensible Reddit/Twitter hook",
            ),
            series=social,
        ),
    }
    return signals
