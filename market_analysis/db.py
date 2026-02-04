"""SQLite database setup and helper functions for sentiment-market tracking."""

import sqlite3
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Optional

DEFAULT_DB_PATH = Path(__file__).parent.parent / "data" / "sentiment_market.db"

# Market session types
MARKET_SESSIONS = ["pre_market", "market_hours", "after_hours", "weekend"]


class Database:
    """SQLite database manager for sentiment and market data."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init_schema(self) -> None:
        """Initialize database schema."""
        with self.connection() as conn:
            conn.executescript(SCHEMA)

    def insert_daily_sentiment(
        self,
        date: str,
        article_count: int,
        avg_score: float,
        positive_count: int,
        negative_count: int,
        neutral_count: int,
        ai_article_count: int = 0,
        ai_avg_score: float = 0.0,
    ) -> None:
        """Insert or replace daily sentiment aggregate."""
        with self.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO daily_sentiment
                (date, article_count, avg_score, positive_count, negative_count,
                 neutral_count, ai_article_count, ai_avg_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    date,
                    article_count,
                    avg_score,
                    positive_count,
                    negative_count,
                    neutral_count,
                    ai_article_count,
                    ai_avg_score,
                ),
            )

    def insert_session_sentiment(
        self,
        trading_date: str,
        market_session: str,
        article_count: int,
        avg_score: float,
        positive_count: int,
        negative_count: int,
        neutral_count: int,
        ai_article_count: int = 0,
        ai_avg_score: float = 0.0,
    ) -> None:
        """Insert or replace session-based sentiment aggregate."""
        with self.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO session_sentiment
                (trading_date, market_session, article_count, avg_score,
                 positive_count, negative_count, neutral_count,
                 ai_article_count, ai_avg_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trading_date,
                    market_session,
                    article_count,
                    avg_score,
                    positive_count,
                    negative_count,
                    neutral_count,
                    ai_article_count,
                    ai_avg_score,
                ),
            )

    def insert_daily_price(
        self,
        date: str,
        ticker: str,
        open_price: float,
        high: float,
        low: float,
        close: float,
        adj_close: float,
        volume: int,
        daily_return: float,
        intraday_return: Optional[float] = None,
    ) -> None:
        """Insert or replace daily stock price."""
        with self.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO daily_prices
                (date, ticker, open, high, low, close, adj_close, volume,
                 daily_return, intraday_return)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (date, ticker, open_price, high, low, close, adj_close, volume,
                 daily_return, intraday_return),
            )

    def insert_article(
        self,
        date: str,
        headline: str,
        source: str,
        sentiment_label: str,
        sentiment_score: float,
        url: str,
        published_at: Optional[str] = None,
        market_session: Optional[str] = None,
        trading_date: Optional[str] = None,
    ) -> None:
        """Insert article if URL doesn't exist."""
        with self.connection() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO articles
                (date, headline, source, sentiment_label, sentiment_score, url,
                 published_at, market_session, trading_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (date, headline, source, sentiment_label, sentiment_score, url,
                 published_at, market_session, trading_date),
            )

    def get_sentiment_with_returns(
        self, ticker: str, lag_days: int = 1
    ) -> list[sqlite3.Row]:
        """Get sentiment data joined with stock returns (with optional lag)."""
        with self.connection() as conn:
            return conn.execute(
                f"""
                SELECT
                    s.date,
                    s.avg_score,
                    s.article_count,
                    s.ai_avg_score,
                    p.daily_return,
                    p.adj_close
                FROM daily_sentiment s
                JOIN daily_prices p
                    ON date(s.date, '+{lag_days} day') = p.date
                WHERE p.ticker = ?
                ORDER BY s.date
                """,
                (ticker,),
            ).fetchall()

    def get_session_sentiment_with_returns(
        self,
        ticker: str,
        market_session: str,
        use_intraday: bool = True,
    ) -> list[sqlite3.Row]:
        """
        Get session-based sentiment joined with appropriate returns.

        For pre_market: same-day intraday returns (open-to-close)
        For after_hours/weekend: next trading day returns
        """
        return_col = "intraday_return" if use_intraday else "daily_return"

        # Pre-market sentiment predicts same-day returns
        if market_session == "pre_market":
            query = f"""
                SELECT
                    s.trading_date,
                    s.avg_score,
                    s.article_count,
                    s.ai_avg_score,
                    p.{return_col} as return_value,
                    p.adj_close
                FROM session_sentiment s
                JOIN daily_prices p ON s.trading_date = p.date
                WHERE p.ticker = ?
                  AND s.market_session = ?
                  AND p.{return_col} IS NOT NULL
                ORDER BY s.trading_date
            """
        # After-hours and weekend predict next trading day
        else:
            query = f"""
                SELECT
                    s.trading_date,
                    s.avg_score,
                    s.article_count,
                    s.ai_avg_score,
                    p.{return_col} as return_value,
                    p.adj_close
                FROM session_sentiment s
                JOIN daily_prices p ON p.date = (
                    SELECT MIN(date) FROM daily_prices
                    WHERE date > s.trading_date AND ticker = ?
                )
                WHERE p.ticker = ?
                  AND s.market_session = ?
                  AND p.{return_col} IS NOT NULL
                ORDER BY s.trading_date
            """

        with self.connection() as conn:
            if market_session == "pre_market":
                return conn.execute(query, (ticker, market_session)).fetchall()
            else:
                return conn.execute(query, (ticker, ticker, market_session)).fetchall()

    def get_date_range(self) -> tuple[Optional[str], Optional[str]]:
        """Get the min and max dates in the sentiment table."""
        with self.connection() as conn:
            row = conn.execute(
                "SELECT MIN(date), MAX(date) FROM daily_sentiment"
            ).fetchone()
            return (row[0], row[1]) if row else (None, None)

    def get_session_date_range(self) -> tuple[Optional[str], Optional[str]]:
        """Get the min and max trading dates in session sentiment table."""
        with self.connection() as conn:
            row = conn.execute(
                "SELECT MIN(trading_date), MAX(trading_date) FROM session_sentiment"
            ).fetchone()
            return (row[0], row[1]) if row else (None, None)

    def get_correlation_data(self, ticker: str) -> list[dict]:
        """Get data formatted for correlation analysis."""
        rows = self.get_sentiment_with_returns(ticker)
        return [
            {
                "date": row["date"],
                "sentiment": row["avg_score"],
                "return": row["daily_return"],
            }
            for row in rows
        ]

    def get_session_correlation_data(
        self,
        ticker: str,
        market_session: str,
        use_intraday: bool = True,
    ) -> list[dict]:
        """Get session-based data formatted for correlation analysis."""
        rows = self.get_session_sentiment_with_returns(
            ticker, market_session, use_intraday
        )
        return [
            {
                "date": row["trading_date"],
                "sentiment": row["avg_score"],
                "return": row["return_value"],
            }
            for row in rows
        ]

    def clear_all_data(self) -> None:
        """Clear all data from tables (for repopulation)."""
        with self.connection() as conn:
            conn.execute("DELETE FROM articles")
            conn.execute("DELETE FROM daily_sentiment")
            conn.execute("DELETE FROM session_sentiment")
            # Keep daily_prices as stock data doesn't need repopulation
            print("Cleared articles, daily_sentiment, and session_sentiment tables")


SCHEMA = """
-- Daily sentiment aggregates from news articles (legacy, kept for compatibility)
CREATE TABLE IF NOT EXISTS daily_sentiment (
    date TEXT PRIMARY KEY,
    article_count INTEGER NOT NULL,
    avg_score REAL NOT NULL,
    positive_count INTEGER NOT NULL,
    negative_count INTEGER NOT NULL,
    neutral_count INTEGER NOT NULL,
    ai_article_count INTEGER DEFAULT 0,
    ai_avg_score REAL DEFAULT 0.0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Session-based sentiment aggregates (pre_market, market_hours, after_hours, weekend)
CREATE TABLE IF NOT EXISTS session_sentiment (
    trading_date TEXT NOT NULL,
    market_session TEXT NOT NULL,
    article_count INTEGER NOT NULL,
    avg_score REAL NOT NULL,
    positive_count INTEGER NOT NULL,
    negative_count INTEGER NOT NULL,
    neutral_count INTEGER NOT NULL,
    ai_article_count INTEGER DEFAULT 0,
    ai_avg_score REAL DEFAULT 0.0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (trading_date, market_session)
);

-- Daily stock prices for tracked tickers
CREATE TABLE IF NOT EXISTS daily_prices (
    date TEXT NOT NULL,
    ticker TEXT NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    adj_close REAL,
    volume INTEGER,
    daily_return REAL,
    intraday_return REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (date, ticker)
);

-- Raw articles for detailed analysis
CREATE TABLE IF NOT EXISTS articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    headline TEXT NOT NULL,
    source TEXT,
    sentiment_label TEXT NOT NULL,
    sentiment_score REAL NOT NULL,
    url TEXT UNIQUE,
    published_at TEXT,
    market_session TEXT,
    trading_date TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_sentiment_date ON daily_sentiment(date);
CREATE INDEX IF NOT EXISTS idx_session_sentiment_date ON session_sentiment(trading_date);
CREATE INDEX IF NOT EXISTS idx_session_sentiment_session ON session_sentiment(market_session);
CREATE INDEX IF NOT EXISTS idx_prices_ticker_date ON daily_prices(ticker, date);
CREATE INDEX IF NOT EXISTS idx_articles_date ON articles(date);
CREATE INDEX IF NOT EXISTS idx_articles_trading_date ON articles(trading_date);
CREATE INDEX IF NOT EXISTS idx_articles_session ON articles(market_session);
"""


if __name__ == "__main__":
    # Initialize database when run directly
    db = Database()
    db.init_schema()
    print(f"Database initialized at: {db.db_path}")
