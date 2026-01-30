"""
Sentiment Classifier Module
Uses pre-trained DistilBERT for sentiment analysis.
"""

from typing import List, Dict, Tuple
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
import torch


class SentimentClassifier:
    """
    Sentiment classifier using pre-trained DistilBERT model.

    Uses 'distilbert-base-uncased-finetuned-sst-2-english' for binary sentiment,
    then maps to positive/negative/neutral based on confidence thresholds.
    """

    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        """
        Initialize the sentiment classifier.

        Args:
            model_name: HuggingFace model name for sentiment analysis
        """
        self.model_name = model_name
        self.device = 0 if torch.cuda.is_available() else -1

        print(f"Loading model: {model_name}")
        print(f"Device: {'CUDA' if self.device == 0 else 'CPU'}")

        self.classifier = pipeline(
            "sentiment-analysis",
            model=model_name,
            device=self.device,
            truncation=True,
            max_length=512
        )
        print("Model loaded successfully!")

    def classify_single(self, text: str) -> Tuple[str, float]:
        """
        Classify sentiment of a single text.

        Args:
            text: Text to classify

        Returns:
            Tuple of (sentiment_label, sentiment_score)
            - sentiment_label: 'positive', 'negative', or 'neutral'
            - sentiment_score: Score from -1 (very negative) to 1 (very positive)
        """
        if not text or not text.strip():
            return "neutral", 0.0

        try:
            result = self.classifier(text[:512])[0]
            label = result["label"]
            score = result["score"]

            # Convert binary classification to sentiment score (-1 to 1)
            if label == "POSITIVE":
                sentiment_score = score  # 0.5 to 1.0 -> positive territory
                # Map to -1 to 1 scale: score of 1.0 -> 1.0, score of 0.5 -> 0.0
                normalized_score = (score - 0.5) * 2
            else:  # NEGATIVE
                sentiment_score = 1 - score  # Invert for negative
                # Map to -1 to 1 scale: score of 1.0 -> -1.0, score of 0.5 -> 0.0
                normalized_score = -((score - 0.5) * 2)

            # Determine label based on normalized score
            if normalized_score > 0.2:
                sentiment_label = "positive"
            elif normalized_score < -0.2:
                sentiment_label = "negative"
            else:
                sentiment_label = "neutral"

            return sentiment_label, round(normalized_score, 4)

        except Exception as e:
            print(f"Error classifying text: {e}")
            return "neutral", 0.0

    def classify_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Classify sentiment of multiple texts.

        Args:
            texts: List of texts to classify
            batch_size: Number of texts to process at once
            show_progress: Whether to show progress bar

        Returns:
            List of (sentiment_label, sentiment_score) tuples
        """
        results = []

        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Classifying sentiments", unit="batch")

        for i in iterator:
            batch = texts[i:i + batch_size]
            # Clean batch - replace empty strings with placeholder
            clean_batch = [t[:512] if t and t.strip() else "neutral text" for t in batch]

            try:
                batch_results = self.classifier(clean_batch)

                for j, result in enumerate(batch_results):
                    original_text = batch[j]

                    # Handle empty/missing text
                    if not original_text or not original_text.strip():
                        results.append(("neutral", 0.0))
                        continue

                    label = result["label"]
                    score = result["score"]

                    # Convert to normalized score
                    if label == "POSITIVE":
                        normalized_score = (score - 0.5) * 2
                    else:
                        normalized_score = -((score - 0.5) * 2)

                    # Determine label
                    if normalized_score > 0.2:
                        sentiment_label = "positive"
                    elif normalized_score < -0.2:
                        sentiment_label = "negative"
                    else:
                        sentiment_label = "neutral"

                    results.append((sentiment_label, round(normalized_score, 4)))

            except Exception as e:
                print(f"Error in batch processing: {e}")
                # Add neutral for failed batch
                results.extend([("neutral", 0.0)] * len(batch))

        return results

    def classify_articles(self, articles: List[Dict]) -> List[Dict]:
        """
        Classify sentiment of article headlines.

        Args:
            articles: List of article dictionaries with 'headline' key

        Returns:
            List of article dictionaries with added sentiment_label and sentiment_score
        """
        headlines = [article.get("headline", "") for article in articles]
        sentiments = self.classify_batch(headlines)

        classified_articles = []
        for article, (label, score) in zip(articles, sentiments):
            classified_article = article.copy()
            classified_article["sentiment_label"] = label
            classified_article["sentiment_score"] = score
            classified_articles.append(classified_article)

        return classified_articles


def classify_headlines(headlines: List[str]) -> List[Dict]:
    """
    Convenience function to classify a list of headlines.

    Args:
        headlines: List of headline strings

    Returns:
        List of dictionaries with headline, sentiment_label, and sentiment_score
    """
    classifier = SentimentClassifier()
    results = classifier.classify_batch(headlines)

    return [
        {
            "headline": headline,
            "sentiment_label": label,
            "sentiment_score": score
        }
        for headline, (label, score) in zip(headlines, results)
    ]


if __name__ == "__main__":
    # Test the classifier
    test_headlines = [
        "Apple announces revolutionary new AI-powered iPhone features",
        "Tech layoffs continue as major companies cut thousands of jobs",
        "Microsoft reports quarterly earnings in line with expectations",
        "Breakthrough in quantum computing promises faster drug discovery",
        "Cybersecurity concerns grow as hackers target major infrastructure",
        "Google unveils exciting new AI assistant capabilities",
        "Twitter faces backlash over controversial policy changes",
        "New study shows mixed results for remote work productivity"
    ]

    print("Testing sentiment classifier on sample headlines:\n")
    classifier = SentimentClassifier()

    for headline in test_headlines:
        label, score = classifier.classify_single(headline)
        print(f"Headline: {headline[:60]}...")
        print(f"  Sentiment: {label} (score: {score:+.4f})")
        print()
