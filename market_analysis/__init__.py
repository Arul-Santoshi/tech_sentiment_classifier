"""Market analysis module for sentiment-stock correlation tracking."""

from .db import Database, MARKET_SESSIONS
from .stock_data import StockDataFetcher, DEFAULT_TICKERS
from .market_session import (
    classify_market_session,
    classify_by_date_only,
    get_session_description,
)
from .correlation import (
    load_correlation_data,
    load_session_correlation_data,
    calculate_correlation,
    analyze_lag_effects,
    analyze_session_correlations,
    generate_correlation_report,
)

__all__ = [
    "Database",
    "MARKET_SESSIONS",
    "StockDataFetcher",
    "DEFAULT_TICKERS",
    "classify_market_session",
    "classify_by_date_only",
    "get_session_description",
    "load_correlation_data",
    "load_session_correlation_data",
    "calculate_correlation",
    "analyze_lag_effects",
    "analyze_session_correlations",
    "generate_correlation_report",
]
