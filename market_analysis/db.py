"""SQLite database setup and helper functions for sentiment-market tracking."""

import sqlite3
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Optional

DEFAULT_DB_PATH = Path(__file__).parent.parent / "data" / "sentiment_market.db"


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
    ) -> None:
        """Insert or replace daily stock price."""
        with self.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO daily_prices
                (date, ticker, open, high, low, close, adj_close, volume, daily_return)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (date, ticker, open_price, high, low, close, adj_close, volume, daily_return),
            )

    def insert_article(
        self,
        date: str,
        headline: str,
        source: str,
        sentiment_label: str,
        sentiment_score: float,
        url: str,
    ) -> None:
        """Insert article if URL doesn't exist."""
        with self.connection() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO articles
                (date, headline, source, sentiment_label, sentiment_score, url)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (date, headline, source, sentiment_label, sentiment_score, url),
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

    def get_date_range(self) -> tuple[Optional[str], Optional[str]]:
        """Get the min and max dates in the sentiment table."""
        with self.connection() as conn:
            row = conn.execute(
                "SELECT MIN(date), MAX(date) FROM daily_sentiment"
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


SCHEMA = """
-- Daily sentiment aggregates from news articles
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
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_sentiment_date ON daily_sentiment(date);
CREATE INDEX IF NOT EXISTS idx_prices_ticker_date ON daily_prices(ticker, date);
CREATE INDEX IF NOT EXISTS idx_articles_date ON articles(date);
"""


if __name__ == "__main__":
    # Initialize database when run directly
    db = Database()
    db.init_schema()
    print(f"Database initialized at: {db.db_path}")
