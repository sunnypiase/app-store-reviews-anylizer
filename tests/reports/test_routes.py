import uuid


async def test_get_report_renders_html(client, make_sample):
    sample_id = await make_sample(
        [{"rating": 5, "title": "Great", "content": "Love this app"}]
    )

    response = await client.get(f"/api/v1/reports/{sample_id}")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "App Store Review Report" in response.text
    # A single 5-star positive review yields no complaint candidates.
    assert "No complaint signal" in response.text


async def test_get_report_shows_fallback_source_for_negative_sample(client, make_sample):
    sample_id = await make_sample(
        [
            {"rating": 1, "title": "Billing issue", "content": "Charged twice, awful support"},
            {"rating": 1, "title": "Billing broken", "content": "Charged twice again, terrible"},
        ]
    )

    response = await client.get(f"/api/v1/reports/{sample_id}")

    assert response.status_code == 200
    assert "Local fallback" in response.text


async def test_get_report_not_found_returns_404(client):
    response = await client.get(f"/api/v1/reports/{uuid.uuid4()}")

    assert response.status_code == 404
