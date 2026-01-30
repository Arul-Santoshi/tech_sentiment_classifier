#!/usr/bin/env python3
"""
Tech Sentiment Classifier - Main Pipeline
Fetches technology and AI news articles, classifies sentiment, and generates visualizations.
"""

import os
import sys
import argparse
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

from news_fetcher import NewsFetcher, fetch_tech_news
from sentiment_classifier import SentimentClassifier
from visualization import (
    plot_sentiment_over_time,
    plot_source_sentiment,
    generate_summary_stats,
    print_summary
)


def run_pipeline(
    api_key: str,
    target_articles: int = 1000,
    output_csv: str = "tech_sentiment_results.csv",
    output_plot: str = "sentiment_analysis.png",
    skip_fetch: bool = False,
    input_csv: str = None
) -> pd.DataFrame:
    """
    Run the complete sentiment analysis pipeline.

    Args:
        api_key: NewsAPI API key
        target_articles: Target number of articles to fetch
        output_csv: Path for output CSV file
        output_plot: Path for output visualization
        skip_fetch: Skip fetching and use existing CSV
        input_csv: Path to existing CSV (used with skip_fetch)

    Returns:
        DataFrame with sentiment analysis results
    """
    print("=" * 60)
    print("TECH SENTIMENT CLASSIFIER PIPELINE")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Step 1: Fetch or load articles
    if skip_fetch and input_csv and os.path.exists(input_csv):
        print(f"Loading existing data from: {input_csv}")
        df = pd.read_csv(input_csv)

        # Check if sentiment already classified
        if 'sentiment_label' in df.columns and 'sentiment_score' in df.columns:
            print("Data already contains sentiment classifications.")
            print(f"Loaded {len(df)} articles")
        else:
            print(f"Loaded {len(df)} articles (needs classification)")
    else:
        print("STEP 1: Fetching news articles from NewsAPI")
        print("-" * 40)

        if not api_key:
            print("ERROR: No API key provided!")
            print("Please set NEWSAPI_KEY environment variable or pass --api-key")
            sys.exit(1)

        articles = fetch_tech_news(api_key, target_count=target_articles)

        if not articles:
            print("ERROR: No articles fetched!")
            sys.exit(1)

        df = pd.DataFrame(articles)
        print(f"\nFetched {len(df)} articles")

    # Step 2: Classify sentiment (if not already done)
    if 'sentiment_label' not in df.columns:
        print("\nSTEP 2: Classifying sentiment using DistilBERT")
        print("-" * 40)

        classifier = SentimentClassifier()
        headlines = df['headline'].tolist()
        sentiments = classifier.classify_batch(headlines, show_progress=True)

        df['sentiment_label'] = [s[0] for s in sentiments]
        df['sentiment_score'] = [s[1] for s in sentiments]

        print(f"\nClassified {len(df)} articles")

    # Step 3: Save to CSV
    print(f"\nSTEP 3: Saving results to CSV")
    print("-" * 40)

    # Ensure columns are in correct order
    columns = ['date', 'headline', 'source', 'sentiment_label', 'sentiment_score']
    extra_cols = [c for c in df.columns if c not in columns]
    df = df[columns + extra_cols]

    df.to_csv(output_csv, index=False)
    print(f"Saved to: {output_csv}")

    # Step 4: Generate summary statistics
    print("\nSTEP 4: Generating summary statistics")
    print("-" * 40)

    stats = generate_summary_stats(df)
    print_summary(stats)

    # Step 5: Create visualizations
    print("\nSTEP 5: Creating visualizations")
    print("-" * 40)

    plot_sentiment_over_time(df, output_path=output_plot)

    # Create source sentiment plot if we have enough sources
    if df['source'].nunique() >= 5:
        source_plot = output_plot.replace('.png', '_by_source.png')
        plot_source_sentiment(df, output_path=source_plot)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    return df


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Tech Sentiment Classifier - Analyze sentiment of technology and AI news"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="NewsAPI API key (or set NEWSAPI_KEY env var)"
    )
    parser.add_argument(
        "--articles",
        type=int,
        default=1000,
        help="Target number of articles to fetch (default: 1000)"
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="tech_sentiment_results.csv",
        help="Output CSV file path (default: tech_sentiment_results.csv)"
    )
    parser.add_argument(
        "--output-plot",
        type=str,
        default="sentiment_analysis.png",
        help="Output plot file path (default: sentiment_analysis.png)"
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip fetching and use existing CSV file"
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        help="Input CSV file to use with --skip-fetch"
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Get API key from args or environment
    api_key = args.api_key or os.getenv("NEWSAPI_KEY")

    if not api_key and not args.skip_fetch:
        print("ERROR: NewsAPI key required!")
        print("Provide via --api-key argument or set NEWSAPI_KEY environment variable")
        print("\nGet your free API key at: https://newsapi.org/register")
        sys.exit(1)

    # Run the pipeline
    run_pipeline(
        api_key=api_key,
        target_articles=args.articles,
        output_csv=args.output_csv,
        output_plot=args.output_plot,
        skip_fetch=args.skip_fetch,
        input_csv=args.input_csv or args.output_csv
    )


if __name__ == "__main__":
    main()
