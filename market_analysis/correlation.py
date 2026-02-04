"""Correlation analysis between sentiment and stock market returns."""

from typing import Optional
import pandas as pd
import numpy as np
from scipy import stats

from .db import Database, DEFAULT_DB_PATH, MARKET_SESSIONS


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


def load_session_correlation_data(
    db: Optional[Database] = None,
    ticker: str = "QQQM",
    market_session: str = "pre_market",
    use_intraday: bool = True,
) -> pd.DataFrame:
    """
    Load session-based sentiment and stock data for correlation analysis.

    Args:
        db: Database instance
        ticker: Stock ticker to analyze
        market_session: Session to analyze ('pre_market', 'after_hours', 'weekend')
        use_intraday: Use intraday returns (open-to-close) for pre_market analysis

    Returns:
        DataFrame with sentiment and return data
    """
    db = db or Database()
    rows = db.get_session_sentiment_with_returns(ticker, market_session, use_intraday)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(
        [dict(row) for row in rows],
        columns=["date", "avg_score", "article_count", "ai_avg_score", "return_value", "adj_close"],
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

    # Handle different column names
    if return_col not in df.columns and "return_value" in df.columns:
        return_col = "return_value"

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


def analyze_session_correlations(
    db: Optional[Database] = None,
    ticker: str = "QQQM",
    use_intraday: bool = True,
) -> pd.DataFrame:
    """
    Analyze correlation for each market session.

    Args:
        db: Database instance
        ticker: Stock ticker to analyze
        use_intraday: Use intraday returns for pre_market

    Returns:
        DataFrame with correlation for each session
    """
    db = db or Database()
    results = []

    for session in MARKET_SESSIONS:
        df = load_session_correlation_data(db, ticker, session, use_intraday)
        corr = calculate_correlation(df, return_col="return_value")

        if "error" not in corr:
            results.append({
                "session": session,
                "pearson_r": corr["pearson_r"],
                "pearson_p": corr["pearson_p"],
                "spearman_r": corr["spearman_r"],
                "spearman_p": corr["spearman_p"],
                "n": corr["n"],
                "sentiment_mean": corr["sentiment_mean"],
                "return_mean": corr["return_mean"],
            })
        else:
            results.append({
                "session": session,
                "pearson_r": None,
                "pearson_p": None,
                "spearman_r": None,
                "spearman_p": None,
                "n": corr.get("n", 0),
                "sentiment_mean": None,
                "return_mean": None,
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
        "=" * 70,
        "SENTIMENT-MARKET CORRELATION REPORT",
        "=" * 70,
        "",
    ]

    date_range = db.get_date_range()
    session_range = db.get_session_date_range()
    lines.append(f"Daily sentiment data range: {date_range[0]} to {date_range[1]}")
    lines.append(f"Session sentiment data range: {session_range[0]} to {session_range[1]}")
    lines.append("")

    # Legacy daily analysis
    lines.append("=" * 70)
    lines.append("DAILY SENTIMENT ANALYSIS (Legacy)")
    lines.append("=" * 70)

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
                sig2 = "**" if corr["pearson_p"] < 0.01 else sig
                lines.append(
                    f"  {label}: r={corr['pearson_r']:+.4f}{sig2} "
                    f"(p={corr['pearson_p']:.4f}, n={corr['n']})"
                )

    # Session-based analysis
    lines.append("\n" + "=" * 70)
    lines.append("SESSION-BASED ANALYSIS (Recommended)")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Session mapping:")
    lines.append("  pre_market  → Same-day intraday returns (open-to-close)")
    lines.append("  after_hours → Next trading day returns")
    lines.append("  weekend     → Monday returns")
    lines.append("")

    for ticker in tickers:
        lines.append(f"\n{ticker}")
        lines.append("-" * 40)

        session_df = analyze_session_correlations(db, ticker, use_intraday=True)

        if session_df.empty:
            lines.append("  No session data available")
            continue

        for _, row in session_df.iterrows():
            session = row["session"]
            n = row["n"]

            if pd.isna(row["pearson_r"]):
                lines.append(f"  {session:12}: Insufficient data (n={n})")
            else:
                sig = ""
                if row["pearson_p"] < 0.01:
                    sig = "**"
                elif row["pearson_p"] < 0.05:
                    sig = "*"

                lines.append(
                    f"  {session:12}: r={row['pearson_r']:+.4f}{sig:2} "
                    f"(p={row['pearson_p']:.4f}, n={n:3})"
                )

    # Summary
    lines.append("\n" + "=" * 70)
    lines.append("INTERPRETATION GUIDE")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Significance: * p<0.05, ** p<0.01")
    lines.append("")
    lines.append("Expected strongest signal: pre_market → same-day intraday returns")
    lines.append("  (News published before market open predicting that day's movement)")
    lines.append("")
    lines.append("Correlation strength:")
    lines.append("  |r| < 0.1  : Negligible")
    lines.append("  |r| 0.1-0.3: Weak")
    lines.append("  |r| 0.3-0.5: Moderate")
    lines.append("  |r| > 0.5  : Strong")
    lines.append("")
    lines.append("Note: Financial correlations above 0.1 can be meaningful for trading")
    lines.append("      strategies, even if they appear 'weak' by academic standards.")
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


def export_session_data(
    db: Optional[Database] = None,
    ticker: str = "QQQM",
    market_session: str = "pre_market",
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Export session-based data for plotting.

    Args:
        db: Database instance
        ticker: Stock ticker
        market_session: Session to export
        output_path: Optional CSV output path

    Returns:
        DataFrame ready for visualization
    """
    db = db or Database()
    df = load_session_correlation_data(db, ticker, market_session, use_intraday=True)

    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df["session"] = market_session

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Exported to: {output_path}")

    return df


if __name__ == "__main__":
    import sys

    db = Database()

    # Check if we have data
    date_range = db.get_date_range()
    session_range = db.get_session_date_range()

    if date_range[0] is None and session_range[0] is None:
        print("No data in database. Run the daily pipeline first.")
        print("  python -m market_analysis.daily_runner init")
        print("  python -m market_analysis.daily_runner import tech_sentiment_results.csv")
        print("  python -m market_analysis.daily_runner backfill --days 90")
        sys.exit(1)

    # Generate and print report
    report = generate_correlation_report(db)
    print(report)

    # Show session analysis for QQQM
    print("\nDetailed Session Analysis (QQQM):")
    print("-" * 40)
    session_df = analyze_session_correlations(db, "QQQM", use_intraday=True)
    if not session_df.empty:
        print(session_df.to_string(index=False))
    else:
        print("Insufficient data for session analysis")

    # Show lag analysis
    print("\n\nLag Analysis - Daily Sentiment (QQQM):")
    print("-" * 40)
    lag_df = analyze_lag_effects(db, "QQQM", max_lag=5)
    if not lag_df.empty:
        print(lag_df.to_string(index=False))
    else:
        print("Insufficient data for lag analysis")
