import uuid
from datetime import datetime, timezone

from app.insights.schemas import ActionableInsight, Insight, SentimentDistribution
from app.insights.service import compute_insights
from app.metrics.service import compute_metrics
from app.reports.service import render_report
from app.reviews.schemas import Review, ReviewsSample


def _sample_with_reviews() -> tuple[uuid.UUID, ReviewsSample, list[Review]]:
    sample_id = uuid.uuid4()
    reviews = [
        Review(
            id=uuid.uuid4(),
            date=datetime(2026, 1, 1, tzinfo=timezone.utc),
            user_name="u",
            title="Billing issue",
            content="Customer service never responds, charged twice",
            rating=1,
            app_version="1.0",
        ),
        Review(
            id=uuid.uuid4(),
            date=datetime(2026, 1, 1, tzinfo=timezone.utc),
            user_name="u2",
            title="Great",
            content="This app is fantastic, I use it every day",
            rating=5,
            app_version="1.0",
        ),
    ]
    sample = ReviewsSample(
        id=sample_id,
        app_id=123,
        country_code="us",
        created_at=datetime.now(timezone.utc),
        reviews=reviews,
    )
    return sample_id, sample, reviews


async def test_render_report_includes_metrics_and_insights_and_charts():
    sample_id, sample, reviews = _sample_with_reviews()
    metrics = compute_metrics(sample_id, reviews)
    insight = await compute_insights(sample_id, reviews)

    html = render_report(sample, metrics, insight)

    assert "<html" in html
    assert "App Store Review Report" in html
    assert "data:image/png;base64," in html
    assert f"{metrics.average_rating:.2f}" in html


def _insight_with(
    sample_id: uuid.UUID,
    *,
    executive_summary: str,
    source: str,
    actionable_insights: list[ActionableInsight],
) -> Insight:
    return Insight(
        sample_id=sample_id,
        review_count=2,
        sentiment_distribution=SentimentDistribution(positive=0, neutral=0, negative=2),
        sentiment_rating_disagreement=[],
        negative_keywords=[],
        actionable_insights=actionable_insights,
        executive_summary=executive_summary,
        actionable_insights_source=source,
    )


async def test_render_report_shows_executive_summary_source_and_evidence():
    sample_id, sample, reviews = _sample_with_reviews()
    metrics = compute_metrics(sample_id, reviews)
    insight = _insight_with(
        sample_id,
        executive_summary="Cancellation complaints dominate the negative feedback.",
        source="gemini",
        actionable_insights=[
            ActionableInsight(
                theme="Subscription cancellation",
                evidence_count=2,
                suggestion="Users report unwanted charges. Recommended next step: audit the flow.",
                evidence_review_ids=[review.id for review in reviews],
            )
        ],
    )

    html = render_report(sample, metrics, insight)

    assert "Cancellation complaints dominate the negative feedback." in html
    assert "Gemini analysis" in html
    assert "Subscription cancellation" in html
    assert "Based on 2 review(s)" in html
    # Every evidence review is rendered in full (title + content excerpt).
    for review in reviews:
        assert review.title in html
        assert review.content in html


async def test_render_report_shows_fallback_source_label():
    sample_id, sample, reviews = _sample_with_reviews()
    metrics = compute_metrics(sample_id, reviews)
    insight = await compute_insights(sample_id, reviews)

    html = render_report(sample, metrics, insight)

    assert "Local fallback" in html
    assert insight.executive_summary in html


async def test_render_report_escapes_model_produced_text():
    sample_id, sample, reviews = _sample_with_reviews()
    metrics = compute_metrics(sample_id, reviews)
    insight = _insight_with(
        sample_id,
        executive_summary='<script>alert("summary")</script>',
        source="gemini",
        actionable_insights=[
            ActionableInsight(
                theme='<b>Bold theme</b>',
                evidence_count=1,
                suggestion='<script>alert("suggestion")</script>',
                evidence_review_ids=[reviews[0].id],
            )
        ],
    )

    html = render_report(sample, metrics, insight)

    assert "<script>alert(" not in html
    assert "<b>Bold theme</b>" not in html
    assert "&lt;script&gt;" in html


async def test_render_report_handles_sample_with_no_reviews():
    sample_id = uuid.uuid4()
    sample = ReviewsSample(
        id=sample_id, app_id=123, country_code="us", created_at=datetime.now(timezone.utc), reviews=[]
    )
    metrics = compute_metrics(sample_id, [])
    insight = await compute_insights(sample_id, [])

    html = render_report(sample, metrics, insight)

    assert "<html" in html
    assert "No negative reviews" in html
