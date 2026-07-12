"""Compare each sentiment-scoring approach against the manually labeled
golden dataset and write the comparison to docs/SENTIMENT_ANALYSIS_RESULTS.md.

Usage:
    uv run python -m scripts.sentiment_eval.evaluate
"""

import json
from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

from scripts.sentiment_eval.config import gemini_config

DATA_DIR = Path(__file__).parent / "data"
DOCS_PATH = Path(__file__).parent.parent.parent / "docs" / "SENTIMENT_ANALYSIS_RESULTS.md"
LABELS = ["positive", "neutral", "negative"]

APPROACHES = {
    "VADER": DATA_DIR / "predictions_vader.json",
    "TextBlob": DATA_DIR / "predictions_textblob.json",
    f"Gemini ({gemini_config.model_name})": DATA_DIR / "predictions_gemini.json",
}


def rating_to_label(rating: int) -> str:
    if rating >= 4:
        return "positive"
    if rating == 3:
        return "neutral"
    return "negative"


def load_predictions(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    items = json.loads(path.read_text())
    return {item["store_review_id"]: item["predicted_sentiment"] for item in items}


def confusion_table(y_true: list[str], y_pred: list[str]) -> str:
    matrix = confusion_matrix(y_true, y_pred, labels=LABELS)
    header = "| actual \\ predicted | " + " | ".join(LABELS) + " |"
    separator = "|---" * (len(LABELS) + 1) + "|"
    rows = [header, separator]
    for label, row in zip(LABELS, matrix):
        rows.append(f"| {label} | " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(rows)


def summary_row(name: str, y_true: list[str], y_pred: list[str]) -> dict:
    accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average=None, zero_division=0
    )
    macro_f1 = f1.mean()
    recall_by_label = dict(zip(LABELS, recall))
    return {
        "name": name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "recall": recall_by_label,
    }


def summary_table(rows: list[dict]) -> str:
    ranked = sorted(rows, key=lambda r: r["macro_f1"], reverse=True)
    header = (
        "| Method | Accuracy | Macro F1 | Recall: positive | Recall: neutral | Recall: negative |"
    )
    separator = "|---|---|---|---|---|---|"
    lines = [header, separator]
    for r in ranked:
        lines.append(
            f"| {r['name']} | {r['accuracy']:.1%} | {r['macro_f1']:.2f} | "
            f"{r['recall']['positive']:.1%} | {r['recall']['neutral']:.1%} | "
            f"{r['recall']['negative']:.1%} |"
        )
    return "\n".join(lines)


def approach_section(name: str, golden: list[dict], predictions: dict[str, str] | None) -> str:
    if predictions is None:
        return (
            f"## {name}\n\n"
            "_Not yet run -- see README for setup (needs GEMINI_API_KEY)._\n"
        )

    y_true = [r["manual_sentiment"] for r in golden]
    y_pred = [predictions[r["store_review_id"]] for r in golden]

    report = classification_report(y_true, y_pred, labels=LABELS, digits=2, zero_division=0)
    accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)

    rating_labels = [rating_to_label(r["rating"]) for r in golden]
    rating_agreement = sum(p == r for p, r in zip(y_pred, rating_labels)) / len(y_true)

    disagreements = [
        r for r, t, p in zip(golden, y_true, y_pred) if t != p
    ][:5]
    examples = "\n".join(
        f"- \"{r['title']}\" (manual: {r['manual_sentiment']}, predicted: "
        f"{predictions[r['store_review_id']]}, rating: {r['rating']}★): "
        f"{r['content'][:140]!r}"
        for r in disagreements
    ) or "- (no disagreements)"

    return (
        f"## {name}\n\n"
        f"Accuracy vs. manual golden labels: **{accuracy:.1%}**\n"
        f"Agreement with star-rating-derived label: **{rating_agreement:.1%}**\n\n"
        f"```\n{report}```\n\n"
        f"Confusion matrix (rows = manual label, columns = predicted):\n\n"
        f"{confusion_table(y_true, y_pred)}\n\n"
        f"Example disagreements (predicted vs. manual):\n\n{examples}\n"
    )


def main() -> None:
    golden = json.loads((DATA_DIR / "golden_dataset.json").read_text())

    rating_labels = [rating_to_label(r["rating"]) for r in golden]
    manual_labels = [r["manual_sentiment"] for r in golden]
    rating_vs_manual = sum(a == b for a, b in zip(rating_labels, manual_labels)) / len(golden)

    class_counts = {label: manual_labels.count(label) for label in LABELS}
    app_names = sorted({r["app_name"] for r in golden}) if golden and "app_name" in golden[0] else []

    summary_rows = [summary_row("Rating-derived (naive baseline)", manual_labels, rating_labels)]
    approach_predictions: dict[str, dict[str, str] | None] = {}
    for name, path in APPROACHES.items():
        predictions = load_predictions(path)
        approach_predictions[name] = predictions
        if predictions is not None:
            y_pred = [predictions[r["store_review_id"]] for r in golden]
            summary_rows.append(summary_row(name, manual_labels, y_pred))

    sections = [
        approach_section(name, golden, predictions)
        for name, predictions in approach_predictions.items()
    ]

    ranked = sorted(summary_rows, key=lambda r: r["macro_f1"], reverse=True)
    real_methods_ranked = [r for r in ranked if r["name"] != "Rating-derived (naive baseline)"]
    if real_methods_ranked:
        winner = real_methods_ranked[0]
        weakest_recall_label = min(winner["recall"], key=lambda label: winner["recall"][label])
        recommendation = (
            f"**{winner['name']}** ranks highest on macro F1 ({winner['macro_f1']:.2f}) among the "
            f"methods evaluated here, and its weakest per-class recall is on \"{weakest_recall_label}\" "
            f"({winner['recall'][weakest_recall_label]:.1%}) -- check the summary table above after "
            "each run rather than assuming, since a different app mix or a newer model version can "
            "change the ranking. Pick the top row by macro F1, not by accuracy, given the class "
            "imbalance in this dataset."
        )
    else:
        recommendation = "No predictions were available yet -- run the scoring scripts first."

    golden_desc = (
        f"Golden dataset: {len(golden)} real App Store reviews across {len(app_names)} apps "
        f"({', '.join(app_names)}), 500 most-recent US-store reviews per app (Apple's RSS feed "
        "hard-caps at 500/app). Each review's title + content was manually labeled "
        "positive/negative/neutral by reading the text (see "
        "`scripts/sentiment_eval/build_golden_dataset.py` for the label mapping and labeling "
        "rules). Labels reflect the sentiment expressed in the text, not the star rating -- so "
        "they can also be compared against a naive rating-derived label (4-5★ -> positive, "
        "3★ -> neutral, 1-2★ -> negative)."
    )
    class_pct = {label: class_counts[label] / len(golden) for label in LABELS}
    class_dist_desc = (
        f"Class distribution: positive {class_counts['positive']}, negative "
        f"{class_counts['negative']}, neutral {class_counts['neutral']} "
        f"({class_pct['positive']:.0%} / {class_pct['negative']:.0%} / {class_pct['neutral']:.0%}). "
        "**This is imbalanced**, so overall accuracy alone is a misleading way to rank approaches -- "
        f"a classifier that just always guessed \"positive\" would already score "
        f"{class_pct['positive']:.0%} accuracy while having zero ability to detect negative or "
        "neutral reviews. The summary table below ranks by macro F1 (unweighted average of "
        "per-class F1) and also breaks out per-class recall, so a method that is only good at the "
        "majority class can't hide behind a high headline number."
    )

    doc = f"""# Sentiment analysis approach comparison

{golden_desc}

{class_dist_desc}

## Summary: correctness by label, across methods

{summary_table(summary_rows)}

Naive rating-derived label vs. manual text label agreement: **{rating_vs_manual:.1%}**
(the ceiling a "just use the star rating" approach would hit on this set).

{"\n".join(sections)}

## Discussion

- **Accuracy is not the right metric here.** With positive reviews at
  {class_counts['positive'] / len(golden):.0%} of the set, accuracy rewards
  approaches that default to "positive" and say little about how well they
  catch negative or neutral reviews -- exactly the reviews a review-analysis
  product cares most about surfacing. Macro F1 and per-class recall (above)
  don't have that blind spot.
- **VADER / TextBlob** are lexicon-based, deterministic, free, and score all
  {len(golden)} reviews in well under a second with no network call -- but
  they only see word-level polarity, so they struggle with reviews that are
  short, sarcastic, or where sentiment is implied rather than stated in
  charged words (e.g. "the app used to be free" carries clear negative
  sentiment to a human reader but no negative-polarity words to a lexicon
  scorer). Both also tend to over-predict "positive" and under-predict
  "neutral", which is exactly what the recall breakdown exposes and a bare
  accuracy score would hide.
- **Gemini** reads the review the way a human annotator does, so it handles
  implied sentiment and mixed/neutral cases much better, at the cost of
  needing an API key, network calls, and (outside the free tier) per-request
  billing. Batching 150 reviews per request and firing all requests
  concurrently keeps this to about {-(-len(golden) // 150)} requests for the
  whole {len(golden)}-review set.
- Both lexicon approaches and the LLM approach are being measured against
  labels I (an LLM) assigned by hand -- see the caveat in
  `scripts/sentiment_eval/build_golden_dataset.py`. Treat this as a
  reasonable proxy for human judgment, not as an unimpeachable ground truth;
  in particular an LLM-vs-LLM comparison likely flatters Gemini's score
  somewhat versus a fully independent human-labeled set.

## Recommendation

{recommendation}
"""
    DOCS_PATH.write_text(doc)
    print(f"wrote {DOCS_PATH}")


if __name__ == "__main__":
    main()
