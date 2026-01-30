"""
Visualization Module
Creates visualizations for sentiment analysis results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import Optional
from datetime import datetime


def setup_style():
    """Set up matplotlib style for consistent visualizations."""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")


def plot_sentiment_over_time(
    df: pd.DataFrame,
    output_path: Optional[str] = None,
    figsize: tuple = (14, 10)
) -> None:
    """
    Create comprehensive sentiment visualization over time.

    Args:
        df: DataFrame with columns: date, sentiment_label, sentiment_score
        output_path: Path to save the figure (optional)
        figsize: Figure size tuple
    """
    setup_style()

    # Convert date column to datetime
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Technology & AI News Sentiment Analysis', fontsize=16, fontweight='bold')

    # 1. Daily average sentiment score (line plot)
    ax1 = axes[0, 0]
    daily_sentiment = df.groupby('date')['sentiment_score'].mean().reset_index()
    daily_sentiment = daily_sentiment.sort_values('date')

    ax1.plot(daily_sentiment['date'], daily_sentiment['sentiment_score'],
             color='steelblue', linewidth=2, marker='o', markersize=3, alpha=0.7)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # Add trend line
    if len(daily_sentiment) > 1:
        z = pd.to_numeric(daily_sentiment['date']).values
        coeffs = pd.np.polyfit(z, daily_sentiment['sentiment_score'].values, 1) if hasattr(pd, 'np') else \
                 __import__('numpy').polyfit(z, daily_sentiment['sentiment_score'].values, 1)
        trend = __import__('numpy').poly1d(coeffs)
        ax1.plot(daily_sentiment['date'], trend(z), color='red', linestyle='--',
                 alpha=0.7, label='Trend')
        ax1.legend()

    ax1.set_xlabel('Date')
    ax1.set_ylabel('Average Sentiment Score')
    ax1.set_title('Daily Average Sentiment Score')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Add color bands for positive/negative zones
    ax1.axhspan(0.2, 1, alpha=0.1, color='green', label='Positive zone')
    ax1.axhspan(-1, -0.2, alpha=0.1, color='red', label='Negative zone')
    ax1.set_ylim(-1, 1)

    # 2. Sentiment distribution (pie chart)
    ax2 = axes[0, 1]
    sentiment_counts = df['sentiment_label'].value_counts()
    colors = {'positive': '#4CAF50', 'neutral': '#FFC107', 'negative': '#F44336'}
    pie_colors = [colors.get(label, 'gray') for label in sentiment_counts.index]

    wedges, texts, autotexts = ax2.pie(
        sentiment_counts.values,
        labels=sentiment_counts.index,
        autopct='%1.1f%%',
        colors=pie_colors,
        explode=[0.02] * len(sentiment_counts),
        shadow=True
    )
    ax2.set_title('Overall Sentiment Distribution')

    # 3. Daily sentiment counts (stacked bar chart)
    ax3 = axes[1, 0]
    daily_counts = df.groupby(['date', 'sentiment_label']).size().unstack(fill_value=0)

    # Ensure all sentiment columns exist
    for col in ['positive', 'neutral', 'negative']:
        if col not in daily_counts.columns:
            daily_counts[col] = 0

    daily_counts = daily_counts[['positive', 'neutral', 'negative']]
    daily_counts = daily_counts.sort_index()

    # Resample to weekly if too many days
    if len(daily_counts) > 30:
        daily_counts.index = pd.to_datetime(daily_counts.index)
        daily_counts = daily_counts.resample('W').sum()
        xlabel = 'Week'
    else:
        xlabel = 'Date'

    daily_counts.plot(
        kind='bar',
        stacked=True,
        ax=ax3,
        color=['#4CAF50', '#FFC107', '#F44336'],
        width=0.8
    )
    ax3.set_xlabel(xlabel)
    ax3.set_ylabel('Number of Articles')
    ax3.set_title('Sentiment Counts Over Time')
    ax3.legend(title='Sentiment')

    # Rotate x-axis labels
    tick_labels = [d.strftime('%m/%d') if hasattr(d, 'strftime') else str(d)[:10]
                   for d in daily_counts.index]
    ax3.set_xticklabels(tick_labels, rotation=45, ha='right')

    # 4. Sentiment score distribution (histogram)
    ax4 = axes[1, 1]
    for label, color in colors.items():
        subset = df[df['sentiment_label'] == label]['sentiment_score']
        if len(subset) > 0:
            ax4.hist(subset, bins=30, alpha=0.6, label=label, color=color)

    ax4.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax4.axvline(x=0.2, color='gray', linestyle=':', alpha=0.5)
    ax4.axvline(x=-0.2, color='gray', linestyle=':', alpha=0.5)
    ax4.set_xlabel('Sentiment Score')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Sentiment Score Distribution')
    ax4.legend(title='Sentiment')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")

    plt.show()


def plot_source_sentiment(
    df: pd.DataFrame,
    top_n: int = 15,
    output_path: Optional[str] = None,
    figsize: tuple = (12, 8)
) -> None:
    """
    Create visualization of sentiment by news source.

    Args:
        df: DataFrame with columns: source, sentiment_label, sentiment_score
        top_n: Number of top sources to include
        output_path: Path to save the figure (optional)
        figsize: Figure size tuple
    """
    setup_style()

    # Get top sources by article count
    top_sources = df['source'].value_counts().head(top_n).index.tolist()
    df_filtered = df[df['source'].isin(top_sources)]

    # Calculate average sentiment by source
    source_sentiment = df_filtered.groupby('source').agg({
        'sentiment_score': 'mean',
        'headline': 'count'
    }).rename(columns={'headline': 'article_count'})
    source_sentiment = source_sentiment.sort_values('sentiment_score', ascending=True)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    colors = ['#F44336' if s < -0.1 else '#4CAF50' if s > 0.1 else '#FFC107'
              for s in source_sentiment['sentiment_score']]

    bars = ax.barh(source_sentiment.index, source_sentiment['sentiment_score'],
                   color=colors, alpha=0.8)

    # Add article count labels
    for i, (score, count) in enumerate(zip(source_sentiment['sentiment_score'],
                                           source_sentiment['article_count'])):
        ax.text(score + 0.02 if score >= 0 else score - 0.02,
                i, f'n={count}', va='center',
                ha='left' if score >= 0 else 'right', fontsize=9)

    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    ax.set_xlabel('Average Sentiment Score')
    ax.set_ylabel('News Source')
    ax.set_title(f'Average Sentiment by Top {top_n} News Sources')
    ax.set_xlim(-1, 1)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Source sentiment visualization saved to: {output_path}")

    plt.show()


def generate_summary_stats(df: pd.DataFrame) -> dict:
    """
    Generate summary statistics for the sentiment analysis.

    Args:
        df: DataFrame with sentiment analysis results

    Returns:
        Dictionary with summary statistics
    """
    stats = {
        "total_articles": len(df),
        "date_range": {
            "start": df['date'].min(),
            "end": df['date'].max()
        },
        "sentiment_distribution": df['sentiment_label'].value_counts().to_dict(),
        "sentiment_percentages": (df['sentiment_label'].value_counts(normalize=True) * 100).round(2).to_dict(),
        "average_sentiment_score": round(df['sentiment_score'].mean(), 4),
        "median_sentiment_score": round(df['sentiment_score'].median(), 4),
        "std_sentiment_score": round(df['sentiment_score'].std(), 4),
        "most_positive_headlines": df.nlargest(5, 'sentiment_score')[['headline', 'sentiment_score']].to_dict('records'),
        "most_negative_headlines": df.nsmallest(5, 'sentiment_score')[['headline', 'sentiment_score']].to_dict('records'),
        "unique_sources": df['source'].nunique(),
        "top_sources": df['source'].value_counts().head(10).to_dict()
    }

    return stats


def print_summary(stats: dict) -> None:
    """Print formatted summary statistics."""
    print("\n" + "=" * 60)
    print("SENTIMENT ANALYSIS SUMMARY")
    print("=" * 60)

    print(f"\nTotal Articles Analyzed: {stats['total_articles']}")
    print(f"Date Range: {stats['date_range']['start']} to {stats['date_range']['end']}")
    print(f"Unique Sources: {stats['unique_sources']}")

    print("\n--- Sentiment Distribution ---")
    for label, count in stats['sentiment_distribution'].items():
        pct = stats['sentiment_percentages'][label]
        print(f"  {label.capitalize()}: {count} ({pct}%)")

    print("\n--- Sentiment Scores ---")
    print(f"  Average: {stats['average_sentiment_score']:+.4f}")
    print(f"  Median:  {stats['median_sentiment_score']:+.4f}")
    print(f"  Std Dev: {stats['std_sentiment_score']:.4f}")

    print("\n--- Most Positive Headlines ---")
    for item in stats['most_positive_headlines'][:3]:
        print(f"  [{item['sentiment_score']:+.4f}] {item['headline'][:70]}...")

    print("\n--- Most Negative Headlines ---")
    for item in stats['most_negative_headlines'][:3]:
        print(f"  [{item['sentiment_score']:+.4f}] {item['headline'][:70]}...")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Test with sample data
    import numpy as np

    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    sample_data = {
        'date': np.random.choice(dates, 500),
        'headline': [f"Sample headline {i}" for i in range(500)],
        'source': np.random.choice(['TechCrunch', 'Wired', 'The Verge', 'Ars Technica', 'CNET'], 500),
        'sentiment_label': np.random.choice(['positive', 'neutral', 'negative'], 500, p=[0.4, 0.35, 0.25]),
        'sentiment_score': np.random.uniform(-1, 1, 500)
    }
    df = pd.DataFrame(sample_data)

    # Generate and print summary
    stats = generate_summary_stats(df)
    print_summary(stats)

    # Create visualizations
    plot_sentiment_over_time(df, output_path='sample_sentiment_analysis.png')
