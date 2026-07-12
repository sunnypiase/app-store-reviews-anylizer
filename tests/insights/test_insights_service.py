import uuid
from datetime import datetime, timezone

import pytest

from app.insights.gemini_classifier import GeminiClassificationError
from app.insights.gemini_insight_generator import (
    GeminiInsightDraft,
    GeminiInsightGenerationError,
    GeminiThemeDraft,
)
from app.insights.service import (
    build_complaint_candidates,
    classify_sentiment_vader,
    classify_sentiments,
    compute_insights,
    find_disagreements,
)
from app.reviews.schemas import Review


def _review(*, rating: int, title: str, content: str) -> Review:
    return Review(
        id=uuid.uuid4(),
        date=datetime(2026, 1, 1, tzinfo=timezone.utc),
        user_name="u",
        title=title,
        content=content,
        rating=rating,
        app_version="1.0",
    )


class _FakeGeminiClassifier:
    def __init__(self, sentiment_by_id=None, error: Exception | None = None):
        self._sentiment_by_id = sentiment_by_id or {}
        self._error = error

    async def classify(self, reviews):
        if self._error is not None:
            raise self._error
        return {review.id: self._sentiment_by_id[review.id] for review in reviews}


class _FakeInsightGenerator:
    def __init__(self, draft: GeminiInsightDraft | None = None, error: Exception | None = None):
        self._draft = draft
        self._error = error
        self.calls: list[list[Review]] = []

    async def generate(self, reviews):
        self.calls.append(list(reviews))
        if self._error is not None:
            raise self._error
        return self._draft


def _theme_draft(evidence_ids, theme="Subscription cancellation") -> GeminiThemeDraft:
    return GeminiThemeDraft(
        theme=theme,
        problem_summary="Users report charges after attempting to cancel.",
        suggestion="Audit the cancellation flow and add an explicit confirmation.",
        evidence_review_ids=list(evidence_ids),
    )


@pytest.fixture
def paraphrase_cluster_reviews() -> list[Review]:
    """Deterministic fixture from the PRD: three subscription/cancellation
    paraphrases, two login/authentication paraphrases, one isolated feature
    request, two positive reviews, and one high-rating review with negative
    text (a complaint hidden behind a 5-star rating).
    """
    return [
        _review(rating=1, title="Charged after cancel", content="I canceled my subscription but was charged again, this is terrible"),
        _review(rating=2, title="Cannot cancel", content="There is no way to cancel the subscription, awful billing experience"),
        _review(rating=1, title="Refund refused", content="Cancellation did not work and support refuses a refund, horrible"),
        _review(rating=1, title="Login broken", content="The app rejects my password every time, login is completely broken"),
        _review(rating=2, title="Sign in fails", content="Sign in fails with an error after the update, cannot authenticate at all"),
        _review(rating=3, title="Dark mode please", content="Would be nice to have a dark mode option someday"),
        _review(rating=5, title="Great app", content="I love this app, it works great and is fantastic"),
        _review(rating=5, title="Excellent", content="Excellent experience, wonderful design, highly recommend"),
        _review(rating=5, title="Good but billing", content="Nice app but they keep charging me after I canceled, awful billing"),
    ]


def test_classify_sentiment_vader_positive_and_negative():
    positive = _review(rating=5, title="Amazing", content="I love this app, it's fantastic and works great")
    negative = _review(rating=1, title="Terrible", content="Worst app ever, awful experience, I hate it")

    assert classify_sentiment_vader(positive) == "positive"
    assert classify_sentiment_vader(negative) == "negative"


async def test_classify_sentiments_uses_gemini_when_available():
    review = _review(rating=1, title="fine", content="fine")
    fake = _FakeGeminiClassifier(sentiment_by_id={review.id: "negative"})

    sentiments = await classify_sentiments([review], fake)

    assert sentiments == ["negative"]


async def test_classify_sentiments_falls_back_to_vader_when_gemini_fails():
    positive = _review(rating=5, title="Amazing", content="I love this app, it's fantastic and works great")
    fake = _FakeGeminiClassifier(error=GeminiClassificationError("boom"))

    sentiments = await classify_sentiments([positive], fake)

    assert sentiments == ["positive"]


async def test_classify_sentiments_uses_vader_when_no_gemini_classifier():
    positive = _review(rating=5, title="Amazing", content="I love this app, it's fantastic and works great")

    sentiments = await classify_sentiments([positive], None)

    assert sentiments == ["positive"]


def test_find_disagreements_flags_high_rating_negative_text():
    review = _review(
        rating=5, title="ok", content="Great app but they keep charging me even after I canceled, awful"
    )

    disagreements = find_disagreements([review], ["negative"])

    assert len(disagreements) == 1
    assert disagreements[0].review_id == review.id
    assert disagreements[0].rating == 5
    assert disagreements[0].sentiment == "negative"


def test_find_disagreements_ignores_matching_rating_and_sentiment():
    review = _review(rating=5, title="ok", content="ok")

    assert find_disagreements([review], ["positive"]) == []


async def test_compute_insights_extracts_keywords_from_negative_reviews_only():
    reviews = [
        _review(
            rating=1, title="Billing issue", content="Customer service never responds, charged twice"
        ),
        _review(
            rating=2,
            title="Cancel subscription",
            content="Cancel subscription is impossible, customer service ignored me",
        ),
        _review(rating=5, title="Great", content="This app is fantastic, I use it every day"),
    ]

    insight = await compute_insights(uuid.uuid4(), reviews)

    assert insight.review_count == 3
    assert len(insight.negative_keywords) > 0
    phrases = {kw.phrase for kw in insight.negative_keywords}
    assert "customer service" in phrases
    # Actionable insight themes must not overlap (no unigram alongside its own bigram).
    themes = [insight_.theme for insight_ in insight.actionable_insights]
    assert len(themes) == len(set(themes))


async def test_negative_keywords_exclude_contraction_fragments():
    reviews = [
        _review(rating=1, title="Bad", content="I don't like it, don't waste your money on this"),
        _review(rating=1, title="Awful", content="Don't bother, it doesn't work and I can't login"),
        _review(rating=5, title="Great", content="This app is fantastic, I use it every day"),
    ]

    insight = await compute_insights(uuid.uuid4(), reviews)

    phrases = {kw.phrase for kw in insight.negative_keywords}
    # "don't"/"doesn't"/"can't" must not leak fragments or contraction tokens.
    assert not phrases & {"don", "doesn", "isn", "dont", "doesnt", "cant"}
    assert "money" in phrases


async def test_negative_keywords_group_inflectional_variants():
    reviews = [
        _review(rating=1, title="Charged twice", content="They charged me twice, unacceptable"),
        _review(rating=1, title="Keeps charging", content="It keeps charging my card without consent"),
        _review(rating=2, title="Charge dispute", content="A surprise charge showed up on my card"),
        _review(rating=5, title="Great", content="This app is fantastic, I use it every day"),
    ]

    insight = await compute_insights(uuid.uuid4(), reviews)

    charge_variants = [
        kw for kw in insight.negative_keywords if kw.phrase in {"charge", "charged", "charging", "charges"}
    ]
    # One merged entry counting every review mentioning any variant.
    assert len(charge_variants) == 1
    assert charge_variants[0].count == 3


async def test_fallback_insights_expose_all_evidence_review_ids():
    reviews = [
        _review(rating=1, title="Billing", content="Charged twice by customer service, awful"),
        _review(rating=1, title="Billing again", content="Customer service keeps charging me"),
        _review(rating=2, title="Support silent", content="Customer service never responds at all"),
        _review(rating=2, title="No refund", content="Customer service refused my refund request"),
    ]

    insight = await compute_insights(uuid.uuid4(), reviews)

    assert insight.actionable_insights_source == "rule_based_fallback"
    [top, *_rest] = [a for a in insight.actionable_insights if a.theme == "Customer Service"]
    assert top.evidence_count == 4
    assert set(top.evidence_review_ids) == {review.id for review in reviews}


async def test_compute_insights_empty_sample():
    insight = await compute_insights(uuid.uuid4(), [])

    assert insight.review_count == 0
    assert insight.negative_keywords == []
    assert insight.actionable_insights == []
    assert insight.sentiment_distribution.positive == 0
    assert insight.sentiment_distribution.neutral == 0
    assert insight.sentiment_distribution.negative == 0
    assert insight.actionable_insights_source == "none"
    assert insight.executive_summary


def test_build_complaint_candidates_is_union_of_rating_and_sentiment():
    low_rating_positive_text = _review(rating=1, title="ok", content="actually fine")
    high_rating_negative_text = _review(rating=5, title="bad", content="keeps crashing")
    happy = _review(rating=5, title="great", content="great")

    candidates = build_complaint_candidates(
        [low_rating_positive_text, high_rating_negative_text, happy],
        ["positive", "negative", "positive"],
    )

    assert candidates == [low_rating_positive_text, high_rating_negative_text]


def test_build_complaint_candidates_deduplicates_reviews_matching_both_signals():
    low_rating_negative_text = _review(rating=1, title="bad", content="terrible")

    candidates = build_complaint_candidates([low_rating_negative_text], ["negative"])

    assert candidates == [low_rating_negative_text]


async def test_compute_insights_negative_keywords_still_use_low_ratings_only(
    paraphrase_cluster_reviews,
):
    # The 5-star review with negative text mentions "billing"; it may feed the
    # complaint candidates but must not feed the rating-only keyword corpus.
    reviews = paraphrase_cluster_reviews
    sentiment_by_id = {review.id: "negative" if review.rating <= 2 else "positive" for review in reviews}
    sentiment_by_id[reviews[-1].id] = "negative"  # high-rating review with negative text
    classifier = _FakeGeminiClassifier(sentiment_by_id=sentiment_by_id)
    generator = _FakeInsightGenerator(
        draft=GeminiInsightDraft(
            executive_summary="Billing dominates.",
            themes=[_theme_draft([reviews[0].id, reviews[1].id])],
        )
    )

    insight = await compute_insights(uuid.uuid4(), reviews, classifier, generator)

    # Keyword extraction saw only the five rating<=2 reviews, none of which
    # contain "recommend" (from the positive reviews) or "dark mode".
    phrases = {kw.phrase for kw in insight.negative_keywords}
    assert not any("recommend" in phrase for phrase in phrases)
    assert not any("dark" in phrase for phrase in phrases)
    # The generator, by contrast, received the union including the 5-star
    # negative-text review.
    [candidates] = generator.calls
    assert reviews[-1] in candidates
    assert len(candidates) == 6


async def test_compute_insights_gemini_success_returns_gemini_source(paraphrase_cluster_reviews):
    reviews = paraphrase_cluster_reviews
    cancellation_ids = [reviews[0].id, reviews[1].id, reviews[2].id, reviews[8].id]
    login_ids = [reviews[3].id, reviews[4].id]
    generator = _FakeInsightGenerator(
        draft=GeminiInsightDraft(
            executive_summary="Cancellation and login complaints dominate the negative feedback.",
            themes=[
                _theme_draft(login_ids, theme="Login failures"),
                _theme_draft(cancellation_ids, theme="Subscription cancellation"),
            ],
        )
    )

    insight = await compute_insights(uuid.uuid4(), reviews, None, generator)

    assert insight.actionable_insights_source == "gemini"
    assert insight.executive_summary == (
        "Cancellation and login complaints dominate the negative feedback."
    )
    # Sorted by evidence count descending; app-computed counts, and every
    # supporting review id is exposed — not a truncated sample.
    assert [a.theme for a in insight.actionable_insights] == [
        "Subscription cancellation",
        "Login failures",
    ]
    assert insight.actionable_insights[0].evidence_count == 4
    assert insight.actionable_insights[0].evidence_review_ids == cancellation_ids
    assert insight.actionable_insights[1].evidence_count == 2
    assert "Recommended next step:" in insight.actionable_insights[0].suggestion


async def test_compute_insights_sorts_equal_evidence_themes_by_name():
    reviews = [_review(rating=1, title=f"bad {i}", content="terrible") for i in range(4)]
    generator = _FakeInsightGenerator(
        draft=GeminiInsightDraft(
            executive_summary="Two equally recurring complaints.",
            themes=[
                _theme_draft([reviews[0].id, reviews[1].id], theme="Zebra crashes"),
                _theme_draft([reviews[2].id, reviews[3].id], theme="Login failures"),
            ],
        )
    )

    insight = await compute_insights(uuid.uuid4(), reviews, None, generator)

    assert [a.theme for a in insight.actionable_insights] == ["Login failures", "Zebra crashes"]


async def test_compute_insights_no_candidates_skips_gemini_and_returns_none_source():
    reviews = [
        _review(rating=5, title="Great", content="I love this app, it's fantastic"),
        _review(rating=4, title="Nice", content="Works great, very happy with it"),
    ]
    generator = _FakeInsightGenerator(
        draft=GeminiInsightDraft(executive_summary="unused", themes=[])
    )

    insight = await compute_insights(uuid.uuid4(), reviews, None, generator)

    assert generator.calls == []
    assert insight.actionable_insights_source == "none"
    assert insight.actionable_insights == []
    assert "2 reviews" in insight.executive_summary


async def test_compute_insights_without_generator_uses_rule_based_fallback():
    reviews = [
        _review(rating=1, title="Billing issue", content="Customer service never responds, charged twice"),
        _review(rating=2, title="Cancel subscription", content="Cancel subscription impossible, customer service ignored me"),
    ]

    insight = await compute_insights(uuid.uuid4(), reviews, None, None)

    assert insight.actionable_insights_source == "rule_based_fallback"
    assert insight.actionable_insights
    assert "local fallback" in insight.executive_summary


async def test_compute_insights_generator_failure_uses_rule_based_fallback():
    reviews = [
        _review(rating=1, title="Billing issue", content="Customer service never responds, charged twice"),
        _review(rating=2, title="Cancel subscription", content="Cancel subscription impossible, customer service ignored me"),
    ]
    generator = _FakeInsightGenerator(error=GeminiInsightGenerationError("boom"))

    insight = await compute_insights(uuid.uuid4(), reviews, None, generator)

    assert insight.actionable_insights_source == "rule_based_fallback"
    assert insight.actionable_insights
    assert "local fallback" in insight.executive_summary


async def test_compute_insights_uses_gemini_classifier_when_provided():
    # A 5-star review VADER would score positive, but Gemini's mock label
    # (negative) wins -- proves compute_insights defers to Gemini, not VADER,
    # whenever a classifier is passed in.
    review = _review(rating=5, title="Great", content="Great app, love it")
    fake = _FakeGeminiClassifier(sentiment_by_id={review.id: "negative"})

    insight = await compute_insights(uuid.uuid4(), [review], fake)

    assert insight.sentiment_distribution.negative == 1
    assert insight.sentiment_distribution.positive == 0
