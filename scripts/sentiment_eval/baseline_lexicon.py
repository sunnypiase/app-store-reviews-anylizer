"""Score the golden dataset with two lexicon-based "default" NLP approaches:
VADER and TextBlob. Neither needs training data or network access.

Threshold conventions (documented, not tuned on the golden set -- these are
each library's own commonly cited defaults, kept as-is to represent
out-of-the-box usage rather than a fitted classifier):

- VADER: score on `title + ". " + content` with SentimentIntensityAnalyzer,
  bucket the `compound` score using VADER's own documented thresholds:
  >= 0.05 -> positive, <= -0.05 -> negative, else neutral.
- TextBlob: score the same text with its default PatternAnalyzer, bucket
  `polarity` with the commonly used +/-0.1 deadband:
  > 0.1 -> positive, < -0.1 -> negative, else neutral.

Usage:
    uv run python -m scripts.sentiment_eval.baseline_lexicon
"""

import json
from pathlib import Path

from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

DATA_DIR = Path(__file__).parent / "data"
GOLDEN_PATH = DATA_DIR / "golden_dataset.json"


def review_text(review: dict) -> str:
    return f"{review['title']}. {review['content']}"


def vader_label(analyzer: SentimentIntensityAnalyzer, text: str) -> str:
    compound = analyzer.polarity_scores(text)["compound"]
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"


def textblob_label(text: str) -> str:
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return "positive"
    if polarity < -0.1:
        return "negative"
    return "neutral"


def main() -> None:
    reviews = json.loads(GOLDEN_PATH.read_text())
    analyzer = SentimentIntensityAnalyzer()

    vader_predictions = []
    textblob_predictions = []
    for review in reviews:
        text = review_text(review)
        vader_predictions.append(
            {"store_review_id": review["store_review_id"], "predicted_sentiment": vader_label(analyzer, text)}
        )
        textblob_predictions.append(
            {"store_review_id": review["store_review_id"], "predicted_sentiment": textblob_label(text)}
        )

    (DATA_DIR / "predictions_vader.json").write_text(json.dumps(vader_predictions, indent=2))
    (DATA_DIR / "predictions_textblob.json").write_text(json.dumps(textblob_predictions, indent=2))
    print(f"wrote {len(vader_predictions)} VADER predictions and {len(textblob_predictions)} TextBlob predictions")


if __name__ == "__main__":
    main()
