"""Stock data fetching using yfinance."""

from datetime import date, datetime, timedelta
from typing import Optional

import yfinance as yf
import pandas as pd

from .db import Database

# Default tickers to track
DEFAULT_TICKERS = ["QQQM", "VOO", "VGT"]


class StockDataFetcher:
    """Fetches and stores stock market data."""

    def __init__(self, db: Optional[Database] = None, tickers: Optional[list[str]] = None):
        self.db = db or Database()
        self.tickers = tickers or DEFAULT_TICKERS

    def fetch_and_store(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch stock data and store in database.

        Args:
            start_date: Start date (YYYY-MM-DD). Defaults to 7 days ago.
            end_date: End date (YYYY-MM-DD). Defaults to today.

        Returns:
            DataFrame with fetched data.
        """
        if end_date is None:
            end_date = date.today().isoformat()
        if start_date is None:
            start_date = (date.today() - timedelta(days=7)).isoformat()

        # Fetch data from yfinance
        df = yf.download(
            self.tickers,
            start=start_date,
            end=end_date,
            auto_adjust=False,
            progress=False,
        )

        if df.empty:
            print(f"No data returned for {self.tickers} between {start_date} and {end_date}")
            return df

        # Process and store each ticker
        for ticker in self.tickers:
            self._store_ticker_data(df, ticker)

        return df

    def _store_ticker_data(self, df: pd.DataFrame, ticker: str) -> None:
        """Extract and store data for a single ticker."""
        try:
            # Handle both single and multi-ticker DataFrames
            if len(self.tickers) == 1:
                ticker_df = df.copy()
            else:
                ticker_df = df.xs(ticker, level=1, axis=1) if isinstance(df.columns, pd.MultiIndex) else df

            # Calculate daily returns (close-to-close) and intraday returns (open-to-close)
            ticker_df = ticker_df.copy()
            ticker_df["Daily_Return"] = ticker_df["Adj Close"].pct_change() * 100
            # Intraday return: (Close - Open) / Open * 100
            ticker_df["Intraday_Return"] = (
                (ticker_df["Close"] - ticker_df["Open"]) / ticker_df["Open"] * 100
            )

            for idx, row in ticker_df.iterrows():
                if pd.isna(row["Close"]):
                    continue

                date_str = idx.strftime("%Y-%m-%d")

                self.db.insert_daily_price(
                    date=date_str,
                    ticker=ticker,
                    open_price=float(row["Open"]) if not pd.isna(row["Open"]) else 0.0,
                    high=float(row["High"]) if not pd.isna(row["High"]) else 0.0,
                    low=float(row["Low"]) if not pd.isna(row["Low"]) else 0.0,
                    close=float(row["Close"]) if not pd.isna(row["Close"]) else 0.0,
                    adj_close=float(row["Adj Close"]) if not pd.isna(row["Adj Close"]) else 0.0,
                    volume=int(row["Volume"]) if not pd.isna(row["Volume"]) else 0,
                    daily_return=float(row["Daily_Return"]) if not pd.isna(row["Daily_Return"]) else 0.0,
                    intraday_return=float(row["Intraday_Return"]) if not pd.isna(row["Intraday_Return"]) else None,
                )

            print(f"Stored {len(ticker_df)} days of data for {ticker}")

        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    def backfill(self, days: int = 90) -> pd.DataFrame:
        """
        Backfill historical data.

        Args:
            days: Number of days to backfill.

        Returns:
            DataFrame with fetched data.
        """
        end_date = date.today().isoformat()
        start_date = (date.today() - timedelta(days=days)).isoformat()

        print(f"Backfilling {days} days of data from {start_date} to {end_date}")
        return self.fetch_and_store(start_date=start_date, end_date=end_date)

    def get_latest_prices(self) -> dict[str, dict]:
        """Get the most recent prices for all tickers."""
        result = {}
        for ticker in self.tickers:
            stock = yf.Ticker(ticker)
            info = stock.info
            result[ticker] = {
                "price": info.get("regularMarketPrice"),
                "change": info.get("regularMarketChangePercent"),
                "name": info.get("shortName"),
            }
        return result


if __name__ == "__main__":
    # Test fetching
    db = Database()
    db.init_schema()

    fetcher = StockDataFetcher(db)
    print("Fetching last 7 days of stock data...")
    df = fetcher.fetch_and_store()
    print(f"\nFetched data shape: {df.shape}")
