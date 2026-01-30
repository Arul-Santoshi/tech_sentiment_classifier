"""
News Fetcher Module
Fetches technology and AI news articles from NewsAPI.
"""

import os
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time


class NewsFetcher:
    """Fetches news articles from NewsAPI."""

    BASE_URL = "https://newsapi.org/v2/everything"

    def __init__(self, api_key: str):
        """
        Initialize the news fetcher.

        Args:
            api_key: NewsAPI API key
        """
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({"X-Api-Key": api_key})

    def fetch_articles(
        self,
        keywords: List[str],
        days_back: int = 90,
        target_count: int = 1000,
        page_size: int = 100
    ) -> List[Dict]:
        """
        Fetch news articles matching the given keywords.

        Args:
            keywords: List of keywords to search for
            days_back: Number of days to look back (max 30 for free tier)
            target_count: Target number of articles to fetch
            page_size: Number of articles per API request (max 100)

        Returns:
            List of article dictionaries
        """
        articles = []
        query = " OR ".join(keywords)

        # NewsAPI free tier only allows 30 days back
        # For paid tier, we can go back further
        from_date = (datetime.now() - timedelta(days=min(days_back, 30))).strftime("%Y-%m-%d")
        to_date = datetime.now().strftime("%Y-%m-%d")

        print(f"Fetching articles from {from_date} to {to_date}")
        print(f"Query: {query}")
        print(f"Target: {target_count} articles")

        page = 1
        max_pages = (target_count // page_size) + 1

        while len(articles) < target_count and page <= max_pages:
            params = {
                "q": query,
                "from": from_date,
                "to": to_date,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": min(page_size, 100),
                "page": page
            }

            try:
                response = self.session.get(self.BASE_URL, params=params)
                response.raise_for_status()
                data = response.json()

                if data.get("status") != "ok":
                    print(f"API Error: {data.get('message', 'Unknown error')}")
                    break

                fetched_articles = data.get("articles", [])
                if not fetched_articles:
                    print("No more articles available")
                    break

                articles.extend(fetched_articles)
                total_results = data.get("totalResults", 0)
                print(f"Page {page}: Fetched {len(fetched_articles)} articles "
                      f"(Total: {len(articles)}/{min(target_count, total_results)})")

                # Check if we've fetched all available articles
                if len(articles) >= total_results:
                    break

                page += 1
                # Rate limiting - be respectful to the API
                time.sleep(0.5)

            except requests.exceptions.RequestException as e:
                print(f"Request error on page {page}: {e}")
                break

        print(f"\nTotal articles fetched: {len(articles)}")
        return articles[:target_count]

    def extract_article_data(self, articles: List[Dict]) -> List[Dict]:
        """
        Extract relevant data from articles.

        Args:
            articles: List of raw article dictionaries from API

        Returns:
            List of cleaned article dictionaries
        """
        cleaned_articles = []

        for article in articles:
            # Skip articles with missing essential data
            if not article.get("title") or article["title"] == "[Removed]":
                continue

            published_at = article.get("publishedAt", "")
            if published_at:
                try:
                    date = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
                    date_str = date.strftime("%Y-%m-%d")
                except ValueError:
                    date_str = published_at[:10] if len(published_at) >= 10 else ""
            else:
                date_str = ""

            cleaned_articles.append({
                "date": date_str,
                "headline": article.get("title", "").strip(),
                "source": article.get("source", {}).get("name", "Unknown"),
                "description": article.get("description", ""),
                "url": article.get("url", "")
            })

        return cleaned_articles


def fetch_tech_news(api_key: str, target_count: int = 1000) -> List[Dict]:
    """
    Convenience function to fetch technology and AI news.

    Args:
        api_key: NewsAPI API key
        target_count: Target number of articles

    Returns:
        List of article dictionaries
    """
    fetcher = NewsFetcher(api_key)
    raw_articles = fetcher.fetch_articles(
        keywords=["technology", "AI"],
        days_back=90,
        target_count=target_count
    )
    return fetcher.extract_article_data(raw_articles)


if __name__ == "__main__":
    # Test the fetcher
    import os
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv("NEWSAPI_KEY")

    if api_key:
        articles = fetch_tech_news(api_key, target_count=10)
        print(f"\nSample articles:")
        for article in articles[:3]:
            print(f"  - {article['date']}: {article['headline'][:60]}...")
    else:
        print("Please set NEWSAPI_KEY environment variable")
