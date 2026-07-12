import uuid
from datetime import datetime, timezone

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


def test_render_report_includes_metrics_and_insights_and_charts():
    sample_id, sample, reviews = _sample_with_reviews()
    metrics = compute_metrics(sample_id, reviews)
    insight = compute_insights(sample_id, reviews)

    html = render_report(sample, metrics, insight)

    assert "<html" in html
    assert "App Store Review Report" in html
    assert "data:image/png;base64," in html
    assert f"{metrics.average_rating:.2f}" in html


def test_render_report_handles_sample_with_no_reviews():
    sample_id = uuid.uuid4()
    sample = ReviewsSample(
        id=sample_id, app_id=123, country_code="us", created_at=datetime.now(timezone.utc), reviews=[]
    )
    metrics = compute_metrics(sample_id, [])
    insight = compute_insights(sample_id, [])

    html = render_report(sample, metrics, insight)

    assert "<html" in html
    assert "No negative reviews" in html
