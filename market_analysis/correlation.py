"""Correlation analysis between sentiment and stock market returns."""

from typing import Optional
import pandas as pd
import numpy as np
from scipy import stats

from .db import Database, DEFAULT_DB_PATH


def load_correlation_data(
    db: Optional[Database] = None,
    ticker: str = "QQQM",
    lag_days: int = 1,
) -> pd.DataFrame:
    """
    Load sentiment and stock data joined for correlation analysis.

    Args:
        db: Database instance
        ticker: Stock ticker to analyze
        lag_days: Days to lag stock returns (1 = next-day returns)

    Returns:
        DataFrame with sentiment and return data
    """
    db = db or Database()
    rows = db.get_sentiment_with_returns(ticker, lag_days)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(
        [dict(row) for row in rows],
        columns=["date", "avg_score", "article_count", "ai_avg_score", "daily_return", "adj_close"],
    )


def calculate_correlation(
    df: pd.DataFrame,
    sentiment_col: str = "avg_score",
    return_col: str = "daily_return",
) -> dict:
    """
    Calculate correlation metrics between sentiment and returns.

    Args:
        df: DataFrame with sentiment and return columns
        sentiment_col: Column name for sentiment scores
        return_col: Column name for returns

    Returns:
        Dictionary with correlation statistics
    """
    if df.empty or len(df) < 5:
        return {"error": "Insufficient data", "n": len(df)}

    # Drop NaN values
    clean_df = df[[sentiment_col, return_col]].dropna()

    if len(clean_df) < 5:
        return {"error": "Insufficient non-null data", "n": len(clean_df)}

    sentiment = clean_df[sentiment_col]
    returns = clean_df[return_col]

    # Pearson correlation
    pearson_r, pearson_p = stats.pearsonr(sentiment, returns)

    # Spearman correlation (rank-based, more robust)
    spearman_r, spearman_p = stats.spearmanr(sentiment, returns)

    return {
        "n": len(clean_df),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "sentiment_mean": float(sentiment.mean()),
        "sentiment_std": float(sentiment.std()),
        "return_mean": float(returns.mean()),
        "return_std": float(returns.std()),
    }


def analyze_lag_effects(
    db: Optional[Database] = None,
    ticker: str = "QQQM",
    max_lag: int = 5,
) -> pd.DataFrame:
    """
    Analyze correlation at different lag periods.

    Args:
        db: Database instance
        ticker: Stock ticker to analyze
        max_lag: Maximum lag days to test

    Returns:
        DataFrame with correlation at each lag
    """
    db = db or Database()
    results = []

    for lag in range(0, max_lag + 1):
        df = load_correlation_data(db, ticker, lag)
        corr = calculate_correlation(df)

        if "error" not in corr:
            results.append({
                "lag_days": lag,
                "pearson_r": corr["pearson_r"],
                "pearson_p": corr["pearson_p"],
                "spearman_r": corr["spearman_r"],
                "spearman_p": corr["spearman_p"],
                "n": corr["n"],
            })

    return pd.DataFrame(results)


def generate_correlation_report(
    db: Optional[Database] = None,
    tickers: Optional[list[str]] = None,
) -> str:
    """
    Generate a text report of correlation findings.

    Args:
        db: Database instance
        tickers: List of tickers to analyze

    Returns:
        Formatted report string
    """
    db = db or Database()
    tickers = tickers or ["QQQM", "VOO", "VGT"]

    lines = [
        "=" * 60,
        "SENTIMENT-MARKET CORRELATION REPORT",
        "=" * 60,
        "",
    ]

    date_range = db.get_date_range()
    lines.append(f"Data range: {date_range[0]} to {date_range[1]}")
    lines.append("")

    for ticker in tickers:
        lines.append(f"\n{ticker}")
        lines.append("-" * 40)

        # Same-day and next-day correlation
        for lag, label in [(0, "Same-day"), (1, "Next-day")]:
            df = load_correlation_data(db, ticker, lag)
            corr = calculate_correlation(df)

            if "error" in corr:
                lines.append(f"  {label}: {corr['error']} (n={corr['n']})")
            else:
                sig = "*" if corr["pearson_p"] < 0.05 else ""
                lines.append(
                    f"  {label}: r={corr['pearson_r']:+.4f}{sig} "
                    f"(p={corr['pearson_p']:.4f}, n={corr['n']})"
                )

    lines.append("\n" + "=" * 60)
    lines.append("* indicates p < 0.05")
    lines.append("")

    return "\n".join(lines)


def export_for_plotting(
    db: Optional[Database] = None,
    ticker: str = "QQQM",
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Export data formatted for plotting.

    Args:
        db: Database instance
        ticker: Stock ticker
        output_path: Optional CSV output path

    Returns:
        DataFrame ready for visualization
    """
    db = db or Database()
    df = load_correlation_data(db, ticker, lag_days=1)

    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # Add rolling averages
    df["sentiment_7d"] = df["avg_score"].rolling(7, min_periods=1).mean()
    df["return_7d"] = df["daily_return"].rolling(7, min_periods=1).mean()

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Exported to: {output_path}")

    return df


if __name__ == "__main__":
    import sys

    db = Database()

    # Check if we have data
    date_range = db.get_date_range()
    if date_range[0] is None:
        print("No data in database. Run the daily pipeline first.")
        print("  python -m market_analysis.daily_runner init")
        print("  python -m market_analysis.daily_runner import tech_sentiment_results.csv")
        print("  python -m market_analysis.daily_runner backfill --days 90")
        sys.exit(1)

    # Generate and print report
    report = generate_correlation_report(db)
    print(report)

    # Show lag analysis for QQQM
    print("\nLag Analysis (QQQM):")
    print("-" * 40)
    lag_df = analyze_lag_effects(db, "QQQM", max_lag=5)
    if not lag_df.empty:
        print(lag_df.to_string(index=False))
    else:
        print("Insufficient data for lag analysis")
