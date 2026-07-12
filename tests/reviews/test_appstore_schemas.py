from app.reviews.appstore.schemas import AppleLookupError, AppleLookupResponse, AppleReviewsResponse

FULL_ENTRY = {
    "id": {"label": "12345"},
    "author": {"name": {"label": "jappleseed"}},
    "updated": {"label": "2026-01-01T00:00:00-07:00"},
    "title": {"label": "Great app"},
    "content": {"label": "Works well, would recommend."},
    "im:rating": {"label": "5"},
    "im:version": {"label": "2.3.1"},
}


def test_parses_full_entry():
    response = AppleReviewsResponse.model_validate({"feed": {"entry": [FULL_ENTRY]}})
    entry = response.feed.entry[0]
    assert entry.entry_id.label == "12345"
    assert entry.author.name.label == "jappleseed"
    assert entry.rating.label == "5"
    assert entry.version.label == "2.3.1"


def test_missing_entry_key_means_no_reviews_not_an_error():
    response = AppleReviewsResponse.model_validate({"feed": {}})
    assert response.feed.entry == []


def test_lookup_response_parses_result_count():
    parsed = AppleLookupResponse.model_validate({"resultCount": 0, "results": []})
    assert parsed.result_count == 0


def test_lookup_error_parses_structured_error_message():
    parsed = AppleLookupError.model_validate(
        {"errorMessage": "Invalid value(s) for key(s): [country]", "queryParameters": {}}
    )
    assert "country" in parsed.error_message
