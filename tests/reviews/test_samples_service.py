import csv
import io
import uuid
from datetime import datetime, timezone

from app.reviews.models import Review, ReviewSample
from app.reviews.samples_service import render_csv


def test_render_csv_includes_header_and_rows():
    sample = ReviewSample(id=uuid.uuid4(), app_id=123, country_code="us")
    sample.reviews = [
        Review(
            id=uuid.uuid4(),
            store_review_id="1",
            date=datetime(2026, 1, 1, tzinfo=timezone.utc),
            user_name="jappleseed",
            title="Great",
            content="Loved it",
            rating=5,
            app_version="1.0",
        )
    ]

    content = render_csv(sample)
    rows = list(csv.reader(io.StringIO(content)))

    assert rows[0] == ["id", "date", "user_name", "title", "content", "rating", "app_version"]
    assert rows[1][2:] == ["jappleseed", "Great", "Loved it", "5", "1.0"]


def test_render_csv_empty_sample_has_only_header():
    sample = ReviewSample(id=uuid.uuid4(), app_id=123, country_code="us")
    sample.reviews = []

    content = render_csv(sample)

    assert content.strip().count("\n") == 0
    assert "id,date,user_name" in content
