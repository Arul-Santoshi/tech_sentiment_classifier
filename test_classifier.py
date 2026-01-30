#!/usr/bin/env python3
"""
Test Script for Sentiment Classifier
Tests the DistilBERT sentiment classifier on sample technology/AI headlines.
"""

from sentiment_classifier import SentimentClassifier


def test_sample_headlines():
    """Test the sentiment classifier on a variety of sample headlines."""

    # Sample headlines with expected sentiments
    test_cases = [
        # Positive headlines
        ("Apple unveils groundbreaking AI features that will transform iPhone experience", "positive"),
        ("NVIDIA stock soars to record high as AI chip demand skyrockets", "positive"),
        ("Scientists achieve major breakthrough in quantum computing", "positive"),
        ("OpenAI releases powerful new model that outperforms competitors", "positive"),
        ("Tech industry adds 50,000 new jobs in strongest quarter yet", "positive"),

        # Negative headlines
        ("Major tech company announces 10,000 layoffs amid economic uncertainty", "negative"),
        ("Cybersecurity breach exposes millions of user accounts", "negative"),
        ("AI startup fails spectacularly after burning through $100M in funding", "negative"),
        ("Critics warn of dangerous AI systems threatening humanity", "negative"),
        ("Tech stocks plummet as recession fears intensify", "negative"),

        # Neutral headlines
        ("Microsoft to hold annual developer conference next month", "neutral"),
        ("Google updates its search algorithm with new features", "neutral"),
        ("Tech industry reports mixed quarterly earnings results", "neutral"),
        ("New smartphone model scheduled for release in Q4", "neutral"),
        ("Congress to discuss AI regulation in upcoming session", "neutral"),
    ]

    print("=" * 70)
    print("SENTIMENT CLASSIFIER TEST")
    print("=" * 70)
    print("\nLoading DistilBERT model...")

    classifier = SentimentClassifier()

    print("\n" + "-" * 70)
    print("Testing on sample headlines:")
    print("-" * 70)

    correct = 0
    total = len(test_cases)

    results = []

    for headline, expected in test_cases:
        label, score = classifier.classify_single(headline)
        is_correct = label == expected

        if is_correct:
            correct += 1
            status = "✓"
        else:
            status = "✗"

        results.append({
            'headline': headline,
            'expected': expected,
            'predicted': label,
            'score': score,
            'correct': is_correct
        })

        print(f"\n{status} Headline: {headline[:60]}...")
        print(f"   Expected: {expected:10} | Predicted: {label:10} | Score: {score:+.4f}")

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"\nAccuracy: {correct}/{total} ({100*correct/total:.1f}%)")

    # Breakdown by category
    positive_cases = [r for r in results if r['expected'] == 'positive']
    negative_cases = [r for r in results if r['expected'] == 'negative']
    neutral_cases = [r for r in results if r['expected'] == 'neutral']

    print(f"\nPositive headlines: {sum(r['correct'] for r in positive_cases)}/{len(positive_cases)} correct")
    print(f"Negative headlines: {sum(r['correct'] for r in negative_cases)}/{len(negative_cases)} correct")
    print(f"Neutral headlines:  {sum(r['correct'] for r in neutral_cases)}/{len(neutral_cases)} correct")

    # Show misclassifications
    misclassified = [r for r in results if not r['correct']]
    if misclassified:
        print("\n" + "-" * 70)
        print("Misclassified headlines:")
        print("-" * 70)
        for r in misclassified:
            print(f"\n  Headline: {r['headline'][:55]}...")
            print(f"  Expected: {r['expected']} | Got: {r['predicted']} (score: {r['score']:+.4f})")

    print("\n" + "=" * 70)

    return results


def test_custom_headlines(headlines: list):
    """
    Test the classifier on custom headlines.

    Args:
        headlines: List of headline strings to classify
    """
    print("\n" + "=" * 70)
    print("CUSTOM HEADLINE TEST")
    print("=" * 70)

    classifier = SentimentClassifier()

    print("\nClassification results:")
    print("-" * 70)

    for headline in headlines:
        label, score = classifier.classify_single(headline)

        # Color indicator based on sentiment
        if label == "positive":
            indicator = "[+]"
        elif label == "negative":
            indicator = "[-]"
        else:
            indicator = "[~]"

        print(f"\n{indicator} {headline}")
        print(f"    Sentiment: {label:10} | Score: {score:+.4f}")

    print("\n" + "=" * 70)


def interactive_test():
    """Run an interactive testing session."""
    print("\n" + "=" * 70)
    print("INTERACTIVE SENTIMENT CLASSIFIER")
    print("=" * 70)
    print("\nEnter headlines to classify (type 'quit' to exit):")

    classifier = SentimentClassifier()

    while True:
        print()
        headline = input("Headline: ").strip()

        if headline.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        if not headline:
            continue

        label, score = classifier.classify_single(headline)
        print(f"  → Sentiment: {label:10} | Score: {score:+.4f}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "--interactive":
            interactive_test()
        else:
            # Test custom headlines from command line
            headlines = sys.argv[1:]
            test_custom_headlines(headlines)
    else:
        # Run standard test
        test_sample_headlines()
