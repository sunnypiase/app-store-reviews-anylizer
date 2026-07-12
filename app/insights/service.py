import logging
import uuid
from collections import Counter

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from app.insights.gemini_classifier import GeminiClassificationError, GeminiSentimentClassifier
from app.insights.schemas import (
    ActionableInsight,
    Disagreement,
    Insight,
    NegativeKeyword,
    Sentiment,
    SentimentDistribution,
)
from app.reviews import schemas as review_schemas

logger = logging.getLogger(__name__)

_analyzer = SentimentIntensityAnalyzer()

# A review counts as "negative" for keyword/theme mining once its star
# rating drops to this or below — independent of the computed sentiment
# label, per the task's plain-language definition of "negative reviews".
_NEGATIVE_RATING_THRESHOLD = 2
_TOP_KEYWORD_COUNT = 10
_TOP_ACTIONABLE_INSIGHTS = 3
_MAX_EXAMPLE_REVIEWS = 3

# "app"/"apps" carry no discriminative signal in a corpus that is, by
# definition, entirely reviews of one app — generic on top of sklearn's
# English stopword list, unlike e.g. the app's own name, which is left in
# since it can still co-occur with genuinely distinctive complaints.
_STOP_WORDS = list(ENGLISH_STOP_WORDS | {"app", "apps", "application"})


def _review_text(review: review_schemas.Review) -> str:
    return f"{review.title} {review.content}"


def classify_sentiment_vader(review: review_schemas.Review) -> Sentiment:
    """VADER's standard compound-score thresholds for 3-way bucketing."""
    compound = _analyzer.polarity_scores(_review_text(review))["compound"]
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"


async def classify_sentiments(
    reviews: list[review_schemas.Review],
    gemini_classifier: GeminiSentimentClassifier | None,
) -> list[Sentiment]:
    """Gemini, when configured, with an automatic VADER fallback -- keeps
    the endpoint usable without an API key and resilient to Gemini being
    down, at the cost of degraded accuracy (see
    docs/SENTIMENT_ANALYSIS_RESULTS.md for the accuracy gap).
    """
    if gemini_classifier is not None:
        try:
            sentiment_by_id = await gemini_classifier.classify(reviews)
        except GeminiClassificationError:
            logger.warning(
                "Gemini sentiment classification failed, falling back to VADER", exc_info=True
            )
        else:
            return [sentiment_by_id[review.id] for review in reviews]
    return [classify_sentiment_vader(review) for review in reviews]


def compute_sentiment_distribution(sentiments: list[Sentiment]) -> SentimentDistribution:
    counts = Counter(sentiments)
    return SentimentDistribution(
        positive=counts["positive"], neutral=counts["neutral"], negative=counts["negative"]
    )


def find_disagreements(
    reviews: list[review_schemas.Review], sentiments: list[Sentiment]
) -> list[Disagreement]:
    disagreements = []
    for review, sentiment in zip(reviews, sentiments, strict=True):
        is_high_rating_negative_text = review.rating >= 4 and sentiment == "negative"
        is_low_rating_positive_text = review.rating <= 2 and sentiment == "positive"
        if is_high_rating_negative_text or is_low_rating_positive_text:
            disagreements.append(
                Disagreement(
                    review_id=review.id, rating=review.rating, sentiment=sentiment, title=review.title
                )
            )
    return disagreements


def extract_negative_keywords(
    reviews: list[review_schemas.Review],
    negative_reviews: list[review_schemas.Review],
    *,
    top_n: int = _TOP_KEYWORD_COUNT,
) -> list[NegativeKeyword]:
    if not negative_reviews:
        return []
    # Fit IDF against the *whole* sample, not just the negative subset: a
    # term that shows up in every review regardless of rating (the app's own
    # name, "app" itself) should score low, while terms concentrated in
    # negative reviews specifically should score high. Fitting IDF on the
    # negative subset alone can't tell "ubiquitous" from "distinctively
    # negative" apart.
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words=_STOP_WORDS, max_features=500)
    try:
        vectorizer.fit([_review_text(review) for review in reviews])
        matrix = vectorizer.transform([_review_text(review) for review in negative_reviews])
    except ValueError:
        # Empty vocabulary after stopword removal (e.g. very short/all-stopword text).
        return []
    if matrix.nnz == 0:
        return []

    scores = matrix.sum(axis=0).A1
    terms = vectorizer.get_feature_names_out()
    ranked = sorted(zip(terms, scores, strict=True), key=lambda item: item[1], reverse=True)[:top_n]

    term_index = {term: idx for idx, term in enumerate(terms)}
    keywords = []
    for term, _score in ranked:
        occurrence_count = int((matrix[:, term_index[term]] > 0).sum())
        if occurrence_count > 0:
            keywords.append(NegativeKeyword(phrase=term, count=occurrence_count))
    return keywords


def build_actionable_insights(
    negative_reviews: list[review_schemas.Review],
    negative_keywords: list[NegativeKeyword],
    *,
    top_n: int = _TOP_ACTIONABLE_INSIGHTS,
) -> list[ActionableInsight]:
    insights: list[ActionableInsight] = []
    chosen_tokens: set[str] = set()
    # Prefer multi-word phrases as themes: a bigram like "customer service"
    # is more specific than its own component words, which otherwise tend to
    # rank just as high by tf-idf and would otherwise crowd out other themes.
    theme_candidates = sorted(negative_keywords, key=lambda kw: kw.phrase.count(" "), reverse=True)
    for keyword in theme_candidates:
        if len(insights) >= top_n:
            break
        candidate_tokens = set(keyword.phrase.split())
        if candidate_tokens & chosen_tokens:
            continue
        matching = [
            review for review in negative_reviews if keyword.phrase in _review_text(review).lower()
        ]
        if not matching:
            continue
        chosen_tokens |= candidate_tokens
        insights.append(
            ActionableInsight(
                theme=keyword.phrase.title(),
                evidence_count=len(matching),
                suggestion=(
                    f"Multiple negative reviews mention '{keyword.phrase}' — investigate and "
                    "address this recurring issue."
                ),
                example_review_ids=[review.id for review in matching[:_MAX_EXAMPLE_REVIEWS]],
            )
        )
    return insights


async def compute_insights(
    sample_id: uuid.UUID,
    reviews: list[review_schemas.Review],
    gemini_classifier: GeminiSentimentClassifier | None = None,
) -> Insight:
    sentiments = await classify_sentiments(reviews, gemini_classifier)
    negative_reviews = [review for review in reviews if review.rating <= _NEGATIVE_RATING_THRESHOLD]
    negative_keywords = extract_negative_keywords(reviews, negative_reviews)
    return Insight(
        sample_id=sample_id,
        review_count=len(reviews),
        sentiment_distribution=compute_sentiment_distribution(sentiments),
        sentiment_rating_disagreement=find_disagreements(reviews, sentiments),
        negative_keywords=negative_keywords,
        actionable_insights=build_actionable_insights(negative_reviews, negative_keywords),
    )
