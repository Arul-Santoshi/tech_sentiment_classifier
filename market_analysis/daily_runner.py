#!/usr/bin/env python3
"""
Daily Runner - Automated pipeline for sentiment-market correlation tracking.

Fetches news, runs sentiment analysis, pulls stock data, and stores in SQLite.
Designed to run daily via cron or GitHub Actions.
"""

import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional
import argparse

import pandas as pd
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from news_fetcher import NewsFetcher
from sentiment_classifier import SentimentClassifier
from market_analysis.db import Database
from market_analysis.stock_data import StockDataFetcher, DEFAULT_TICKERS

# AI-related keywords for filtering
AI_KEYWORDS = ["AI", "OpenAI", "Nvidia", "Anthropic", "LLM", "GPT", "Codex", "agent", "machine learning"]


def aggregate_sentiment(df: pd.DataFrame) -> dict:
    """
    Aggregate sentiment metrics from a DataFrame of articles.

    Args:
        df: DataFrame with sentiment_label and sentiment_score columns

    Returns:
        Dictionary of aggregated metrics
    """
    ai_mask = df["headline"].str.contains("|".join(AI_KEYWORDS), case=False, na=False)

    return {
        "article_count": len(df),
        "avg_score": float(df["sentiment_score"].mean()),
        "positive_count": int((df["sentiment_label"] == "positive").sum()),
        "negative_count": int((df["sentiment_label"] == "negative").sum()),
        "neutral_count": int((df["sentiment_label"] == "neutral").sum()),
        "ai_article_count": int(ai_mask.sum()),
        "ai_avg_score": float(df.loc[ai_mask, "sentiment_score"].mean()) if ai_mask.any() else 0.0,
    }


def run_daily_pipeline(
    api_key: str,
    db: Optional[Database] = None,
    target_articles: int = 100,
    fetch_days_back: int = 1,
    backfill_stock_days: int = 7,
    tickers: Optional[list[str]] = None,
) -> dict:
    """
    Run the complete daily pipeline.

    Args:
        api_key: NewsAPI API key
        db: Database instance (creates new if None)
        target_articles: Number of articles to fetch
        fetch_days_back: Days of news to fetch
        backfill_stock_days: Days of stock data to backfill
        tickers: Stock tickers to track

    Returns:
        Dictionary with run statistics
    """
    db = db or Database()
    db.init_schema()
    tickers = tickers or DEFAULT_TICKERS

    run_date = date.today().isoformat()
    stats = {"run_date": run_date, "success": False}

    print("=" * 60)
    print(f"DAILY SENTIMENT-MARKET PIPELINE")
    print(f"Run date: {run_date}")
    print("=" * 60)

    # Step 1: Fetch news articles
    print("\n[1/4] Fetching news articles...")
    print("-" * 40)

    try:
        fetcher = NewsFetcher(api_key)
        raw_articles = fetcher.fetch_articles(
            keywords=["technology", "AI"],
            days_back=fetch_days_back,
            target_count=target_articles,
        )
        articles = fetcher.extract_article_data(raw_articles)
        stats["articles_fetched"] = len(articles)
        print(f"Fetched {len(articles)} articles")
    except Exception as e:
        print(f"Error fetching news: {e}")
        stats["error"] = str(e)
        return stats

    if not articles:
        print("No articles fetched, skipping sentiment analysis")
        stats["error"] = "No articles fetched"
        return stats

    # Step 2: Run sentiment classification
    print("\n[2/4] Running sentiment classification...")
    print("-" * 40)

    try:
        classifier = SentimentClassifier()
        headlines = [a["headline"] for a in articles]
        sentiments = classifier.classify_batch(headlines, show_progress=True)

        # Add sentiment to articles
        for article, (label, score) in zip(articles, sentiments):
            article["sentiment_label"] = label
            article["sentiment_score"] = score

        df = pd.DataFrame(articles)
        print(f"Classified {len(df)} articles")
    except Exception as e:
        print(f"Error in classification: {e}")
        stats["error"] = str(e)
        return stats

    # Step 3: Store sentiment data
    print("\n[3/4] Storing sentiment data...")
    print("-" * 40)

    try:
        # Store individual articles
        for _, row in df.iterrows():
            db.insert_article(
                date=row["date"],
                headline=row["headline"],
                source=row["source"],
                sentiment_label=row["sentiment_label"],
                sentiment_score=row["sentiment_score"],
                url=row.get("url", ""),
            )

        # Aggregate by date and store
        for article_date, group in df.groupby("date"):
            agg = aggregate_sentiment(group)
            db.insert_daily_sentiment(
                date=article_date,
                article_count=agg["article_count"],
                avg_score=agg["avg_score"],
                positive_count=agg["positive_count"],
                negative_count=agg["negative_count"],
                neutral_count=agg["neutral_count"],
                ai_article_count=agg["ai_article_count"],
                ai_avg_score=agg["ai_avg_score"],
            )
            print(f"  {article_date}: {agg['article_count']} articles, avg score: {agg['avg_score']:.4f}")

        stats["dates_processed"] = df["date"].nunique()
    except Exception as e:
        print(f"Error storing sentiment: {e}")
        stats["error"] = str(e)
        return stats

    # Step 4: Fetch stock data
    print("\n[4/4] Fetching stock market data...")
    print("-" * 40)

    try:
        stock_fetcher = StockDataFetcher(db, tickers)
        stock_fetcher.fetch_and_store(
            start_date=(date.today() - timedelta(days=backfill_stock_days)).isoformat(),
            end_date=date.today().isoformat(),
        )
        stats["tickers_updated"] = tickers
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        stats["stock_error"] = str(e)
        # Don't fail the whole pipeline for stock data issues

    # Summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED")
    print("=" * 60)

    date_range = db.get_date_range()
    print(f"Database date range: {date_range[0]} to {date_range[1]}")
    print(f"Articles processed: {stats.get('articles_fetched', 0)}")
    print(f"Dates in database: {stats.get('dates_processed', 0)}")

    stats["success"] = True
    return stats


def backfill_historical(
    api_key: str,
    db: Optional[Database] = None,
    stock_days: int = 90,
    tickers: Optional[list[str]] = None,
) -> None:
    """
    Backfill historical stock data (news backfill requires paid API).

    Args:
        api_key: NewsAPI API key (unused for stock backfill, kept for consistency)
        db: Database instance
        stock_days: Days of stock data to backfill
        tickers: Stock tickers to track
    """
    db = db or Database()
    db.init_schema()
    tickers = tickers or DEFAULT_TICKERS

    print("=" * 60)
    print("BACKFILLING HISTORICAL STOCK DATA")
    print("=" * 60)

    stock_fetcher = StockDataFetcher(db, tickers)
    stock_fetcher.backfill(days=stock_days)

    print("\nBackfill complete!")
    print(f"Database location: {db.db_path}")


def import_existing_csv(
    csv_path: str,
    db: Optional[Database] = None,
) -> None:
    """
    Import existing sentiment CSV into the database.

    Args:
        csv_path: Path to CSV file with sentiment data
        db: Database instance
    """
    db = db or Database()
    db.init_schema()

    print(f"Importing from: {csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = ["date", "headline", "sentiment_label", "sentiment_score"]

    if not all(col in df.columns for col in required_cols):
        print(f"Error: CSV must contain columns: {required_cols}")
        return

    # Store individual articles
    for _, row in df.iterrows():
        db.insert_article(
            date=row["date"],
            headline=row["headline"],
            source=row.get("source", "Unknown"),
            sentiment_label=row["sentiment_label"],
            sentiment_score=row["sentiment_score"],
            url=row.get("url", ""),
        )

    # Aggregate by date
    for article_date, group in df.groupby("date"):
        agg = aggregate_sentiment(group)
        db.insert_daily_sentiment(
            date=article_date,
            article_count=agg["article_count"],
            avg_score=agg["avg_score"],
            positive_count=agg["positive_count"],
            negative_count=agg["negative_count"],
            neutral_count=agg["neutral_count"],
            ai_article_count=agg["ai_article_count"],
            ai_avg_score=agg["ai_avg_score"],
        )

    print(f"Imported {len(df)} articles across {df['date'].nunique()} dates")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Daily sentiment-market correlation pipeline"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Daily run command
    run_parser = subparsers.add_parser("run", help="Run daily pipeline")
    run_parser.add_argument("--api-key", help="NewsAPI key (or set NEWSAPI_KEY env var)")
    run_parser.add_argument("--articles", type=int, default=100, help="Target articles to fetch")
    run_parser.add_argument("--days", type=int, default=1, help="Days of news to fetch")

    # Backfill command
    backfill_parser = subparsers.add_parser("backfill", help="Backfill historical stock data")
    backfill_parser.add_argument("--days", type=int, default=90, help="Days to backfill")
    backfill_parser.add_argument(
        "--tickers", nargs="+", default=DEFAULT_TICKERS, help="Tickers to fetch"
    )

    # Import command
    import_parser = subparsers.add_parser("import", help="Import existing CSV")
    import_parser.add_argument("csv_path", help="Path to CSV file")

    # Init command
    subparsers.add_parser("init", help="Initialize database schema")

    args = parser.parse_args()
    load_dotenv()

    db = Database()

    if args.command == "run":
        api_key = args.api_key or os.getenv("NEWSAPI_KEY")
        if not api_key:
            print("Error: NewsAPI key required. Set NEWSAPI_KEY or use --api-key")
            sys.exit(1)
        run_daily_pipeline(api_key, db, args.articles, args.days)

    elif args.command == "backfill":
        backfill_historical(None, db, args.days, args.tickers)

    elif args.command == "import":
        import_existing_csv(args.csv_path, db)

    elif args.command == "init":
        db.init_schema()
        print(f"Database initialized at: {db.db_path}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
