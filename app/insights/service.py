import logging
import re
import uuid
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
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
from app.reviews import schemas as review_schemas

logger = logging.getLogger(__name__)

_analyzer = SentimentIntensityAnalyzer()

# A review counts as "negative" for keyword/theme mining once its star
# rating drops to this or below — independent of the computed sentiment
# label, per the task's plain-language definition of "negative reviews".
_NEGATIVE_RATING_THRESHOLD = 2
_TOP_KEYWORD_COUNT = 10
_TOP_ACTIONABLE_INSIGHTS = 3

# Apostrophes are stripped before tokenization (see _normalize_text), so
# contractions arrive as single tokens ("dont", "cant") instead of the junk
# fragments sklearn's default tokenizer would produce ("don", "t") — these
# apostrophe-less forms then need their own stopword entries.
_CONTRACTION_STOP_WORDS = {
    "aint", "arent", "cant", "cannot", "couldnt", "couldve", "didnt", "doesnt",
    "dont", "hadnt", "hasnt", "havent", "hes", "id", "ill", "im", "isnt",
    "its", "ive", "shes", "shouldnt", "shouldve", "thats", "theres", "theyd",
    "theyll", "theyre", "theyve", "wasnt", "werent", "whats", "wont",
    "wouldnt", "wouldve", "youd", "youll", "youre", "youve",
}

# "app"/"apps" carry no discriminative signal in a corpus that is, by
# definition, entirely reviews of one app — generic on top of sklearn's
# English stopword list, unlike e.g. the app's own name, which is left in
# since it can still co-occur with genuinely distinctive complaints.
_STOP_WORDS = list(ENGLISH_STOP_WORDS | _CONTRACTION_STOP_WORDS | {"app", "apps", "application"})

_WORD_PATTERN = re.compile(r"[a-z]{2,}")


def _review_text(review: review_schemas.Review) -> str:
    return f"{review.title} {review.content}"


def _normalize_text(text: str) -> str:
    """Lowercase and drop apostrophes so "don't"/"don’t" become "dont" rather
    than splitting into meaningless fragments at tokenization time."""
    return text.lower().replace("'", "").replace("’", "")


def _stem(token: str) -> str:
    """Light suffix-stripping stem, just aggressive enough to group
    inflections of one complaint ("charge"/"charged"/"charging"/"charges",
    "canceled"/"cancelled") under a single key. Stems are grouping keys only
    and never shown to users — display always uses a real surface form.
    """
    for suffix in ("ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            token = token[: -len(suffix)]
            break
    if len(token) >= 4 and token[-1] == token[-2]:
        token = token[:-1]
    if len(token) >= 4 and token.endswith("e"):
        token = token[:-1]
    return token


def _stem_phrase(phrase: str) -> str:
    return " ".join(_stem(token) for token in phrase.split())


def _stemmed_tokens(text: str) -> set[str]:
    return {_stem(token) for token in _WORD_PATTERN.findall(_normalize_text(text))}


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
    # Fit IDF against the *whole* sample, not just the negative subset: a
    # term that shows up in every review regardless of rating (the app's own
    # name, "app" itself) should score low, while terms concentrated in
    # negative reviews specifically should score high. Fitting IDF on the
    # negative subset alone can't tell "ubiquitous" from "distinctively
    # negative" apart.
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words=_STOP_WORDS,
        max_features=500,
        preprocessor=_normalize_text,
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
    # Merge inflections ("charge"/"charged"/"charging") into one keyword so
    # the top-N list covers N distinct complaints instead of repeating one;
    # each merged keyword displays its highest-scoring surface form and
    # counts reviews containing *any* variant.
    groups: dict[str, _KeywordGroup] = {}
    for idx, term in enumerate(vectorizer.get_feature_names_out()):
        groups.setdefault(_stem_phrase(term), _KeywordGroup()).add(
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
        (review, _stemmed_tokens(_review_text(review))) for review in negative_reviews
    ]
    # Prefer multi-word phrases as themes: a bigram like "customer service"
    # is more specific than its own component words, which otherwise tend to
    # rank just as high by tf-idf and would otherwise crowd out other themes.
    theme_candidates = sorted(negative_keywords, key=lambda kw: kw.phrase.count(" "), reverse=True)
    for keyword in theme_candidates:
        if len(insights) >= top_n:
            break
        phrase_stems = {_stem(token) for token in keyword.phrase.split()}
        if phrase_stems & chosen_stems:
            continue
        # Stem-based matching so a review saying "charging" counts as
        # evidence for the "charged" keyword and vice versa.
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
    """Union of low-rated reviews and negative-text reviews, deduplicated by
    id (input order preserved). The union keeps the assignment's rating-based
    definition while also catching complaints hidden behind 3-5 star ratings.
    """
    return [
        review
        for review, sentiment in zip(reviews, sentiments, strict=True)
        if review.rating <= _NEGATIVE_RATING_THRESHOLD or sentiment == "negative"
    ]


def _draft_to_actionable_insights(draft: GeminiInsightDraft) -> list[ActionableInsight]:
    """The app — not the LLM — computes evidence counts from the validated
    evidence lists (every cited review is included, not a truncated sample);
    text fields are capped to the public-schema limits.
    """
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
    # negative_keywords keeps its original rating-only population; only the
    # actionable-insight generation widens to the rating/sentiment union.
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
