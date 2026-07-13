import logging
import uuid
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from app.insights.gemini_classifier import GeminiClassificationError, GeminiSentimentClassifier
from app.insights.gemini_insight_generator import (
    GeminiInsightDraft,
    GeminiInsightGenerationError,
    GeminiInsightGenerator,
)
from app.insights.schemas import (
    EXECUTIVE_SUMMARY_MAX_LENGTH,
    SUGGESTION_MAX_LENGTH,
    THEME_MAX_LENGTH,
    ActionableInsight,
    ActionableInsightsSource,
    Disagreement,
    Insight,
    NegativeKeyword,
    Sentiment,
    SentimentDistribution,
)
from app.insights.text_preparation import (
    STOP_WORDS,
    TOKEN_PATTERN,
    normalize_text,
    stem,
    stem_phrase,
    stemmed_tokens,
)
from app.reviews import schemas as review_schemas

logger = logging.getLogger(__name__)

_analyzer = SentimentIntensityAnalyzer()

# Rating at or below this counts as "negative" for keyword mining, regardless of sentiment.
_NEGATIVE_RATING_THRESHOLD = 2
_TOP_KEYWORD_COUNT = 10
_TOP_ACTIONABLE_INSIGHTS = 3


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
    """Gemini when configured, with automatic VADER fallback (degraded accuracy)."""
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


@dataclass
class _KeywordGroup:
    """Inflectional variants of one term, merged under a shared stem key."""

    score: float = 0.0
    display: str = ""
    display_score: float = -1.0
    presence: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))

    def add(self, term: str, score: float, presence: np.ndarray) -> None:
        self.score += score
        self.presence = presence if self.presence.size == 0 else self.presence | presence
        if score > self.display_score:
            self.display = term
            self.display_score = score

    @property
    def count(self) -> int:
        """Number of negative reviews containing any variant of the term."""
        return int(self.presence.sum())


def extract_negative_keywords(
    reviews: list[review_schemas.Review],
    negative_reviews: list[review_schemas.Review],
    *,
    top_n: int = _TOP_KEYWORD_COUNT,
) -> list[NegativeKeyword]:
    if not negative_reviews:
        return []
    # IDF is fit on the whole sample so ubiquitous terms score low; tokenization
    # comes from text_preparation so sklearn sees the same tokens as the pipeline.
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words=STOP_WORDS,
        max_features=500,
        preprocessor=normalize_text,
        token_pattern=TOKEN_PATTERN,
    )
    try:
        vectorizer.fit([_review_text(review) for review in reviews])
        matrix = vectorizer.transform([_review_text(review) for review in negative_reviews])
    except ValueError:
        # Empty vocabulary after stopword removal (e.g. very short/all-stopword text).
        return []
    if matrix.nnz == 0:
        return []

    scores = matrix.sum(axis=0).A1
    presence = (matrix > 0).toarray()
    # Merge inflections under one stem so the top-N list covers N distinct complaints.
    groups: dict[str, _KeywordGroup] = {}
    for idx, term in enumerate(vectorizer.get_feature_names_out()):
        groups.setdefault(stem_phrase(term), _KeywordGroup()).add(
            term, float(scores[idx]), presence[:, idx]
        )
    ranked = sorted(groups.values(), key=lambda group: group.score, reverse=True)[:top_n]
    return [
        NegativeKeyword(phrase=group.display, count=group.count)
        for group in ranked
        if group.count > 0
    ]


def build_actionable_insights(
    negative_reviews: list[review_schemas.Review],
    negative_keywords: list[NegativeKeyword],
    *,
    top_n: int = _TOP_ACTIONABLE_INSIGHTS,
) -> list[ActionableInsight]:
    insights: list[ActionableInsight] = []
    chosen_stems: set[str] = set()
    review_stems = [
        (review, stemmed_tokens(_review_text(review))) for review in negative_reviews
    ]
    # Prefer bigrams as themes — more specific than their component words.
    theme_candidates = sorted(negative_keywords, key=lambda kw: kw.phrase.count(" "), reverse=True)
    for keyword in theme_candidates:
        if len(insights) >= top_n:
            break
        phrase_stems = {stem(token) for token in keyword.phrase.split()}
        if phrase_stems & chosen_stems:
            continue
        # Stem-based matching so "charging" counts as evidence for "charged".
        matching = [review for review, stems in review_stems if phrase_stems <= stems]
        if not matching:
            continue
        chosen_stems |= phrase_stems
        insights.append(
            ActionableInsight(
                theme=keyword.phrase.title(),
                evidence_count=len(matching),
                suggestion=(
                    f"{len(matching)} negative reviews mention '{keyword.phrase}' — investigate "
                    "and address this recurring issue."
                ),
                evidence_review_ids=[review.id for review in matching],
            )
        )
    return insights


def build_complaint_candidates(
    reviews: list[review_schemas.Review], sentiments: list[Sentiment]
) -> list[review_schemas.Review]:
    """Union of low-rated and negative-text reviews, input order preserved."""
    return [
        review
        for review, sentiment in zip(reviews, sentiments, strict=True)
        if review.rating <= _NEGATIVE_RATING_THRESHOLD or sentiment == "negative"
    ]


def _draft_to_actionable_insights(draft: GeminiInsightDraft) -> list[ActionableInsight]:
    """Evidence counts are computed here from validated ids, never taken from the LLM."""
    insights = [
        ActionableInsight(
            theme=theme.theme[:THEME_MAX_LENGTH],
            evidence_count=len(theme.evidence_review_ids),
            suggestion=(f"{theme.problem_summary} Recommended next step: {theme.suggestion}")[
                :SUGGESTION_MAX_LENGTH
            ],
            evidence_review_ids=theme.evidence_review_ids,
        )
        for theme in draft.themes
    ]
    insights.sort(key=lambda insight: (-insight.evidence_count, insight.theme))
    return insights


def _fallback_executive_summary(review_count: int, candidate_count: int) -> str:
    return (
        f"Analyzed {review_count} reviews. {candidate_count} reviews contained a negative "
        "rating or negative text signal; recurring themes were generated with the local fallback."
    )


def _no_complaints_executive_summary(review_count: int) -> str:
    return (
        f"Analyzed {review_count} reviews. No reviews contained a negative rating or "
        "negative text signal, so there are no recurring complaint themes to report."
    )


async def _generate_actionable_insights(
    complaint_candidates: list[review_schemas.Review],
    negative_reviews: list[review_schemas.Review],
    negative_keywords: list[NegativeKeyword],
    insight_generator: GeminiInsightGenerator | None,
    review_count: int,
) -> tuple[list[ActionableInsight], str, ActionableInsightsSource]:
    if not complaint_candidates:
        return [], _no_complaints_executive_summary(review_count), "none"
    if insight_generator is not None:
        try:
            draft = await insight_generator.generate(complaint_candidates)
        except GeminiInsightGenerationError:
            logger.warning(
                "Gemini insight generation failed, falling back to rule-based insights",
                exc_info=True,
            )
        else:
            return (
                _draft_to_actionable_insights(draft),
                draft.executive_summary[:EXECUTIVE_SUMMARY_MAX_LENGTH],
                "gemini",
            )
    return (
        build_actionable_insights(negative_reviews, negative_keywords),
        _fallback_executive_summary(review_count, len(complaint_candidates)),
        "rule_based_fallback",
    )


async def compute_insights(
    sample_id: uuid.UUID,
    reviews: list[review_schemas.Review],
    gemini_classifier: GeminiSentimentClassifier | None = None,
    insight_generator: GeminiInsightGenerator | None = None,
) -> Insight:
    sentiments = await classify_sentiments(reviews, gemini_classifier)
    # negative_keywords stays rating-only; only actionable insights use the wider union.
    negative_reviews = [review for review in reviews if review.rating <= _NEGATIVE_RATING_THRESHOLD]
    negative_keywords = extract_negative_keywords(reviews, negative_reviews)
    complaint_candidates = build_complaint_candidates(reviews, sentiments)
    actionable_insights, executive_summary, source = await _generate_actionable_insights(
        complaint_candidates, negative_reviews, negative_keywords, insight_generator, len(reviews)
    )
    logger.info(
        "Actionable insights computed: source=%s candidates=%d themes=%d",
        source,
        len(complaint_candidates),
        len(actionable_insights),
    )
    return Insight(
        sample_id=sample_id,
        review_count=len(reviews),
        sentiment_distribution=compute_sentiment_distribution(sentiments),
        sentiment_rating_disagreement=find_disagreements(reviews, sentiments),
        negative_keywords=negative_keywords,
        actionable_insights=actionable_insights,
        executive_summary=executive_summary,
        actionable_insights_source=source,
    )
