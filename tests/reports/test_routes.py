import uuid


async def test_get_report_renders_html(client, make_sample):
    sample_id = await make_sample(
        [{"rating": 5, "title": "Great", "content": "Love this app"}]
    )

    response = await client.get(f"/api/v1/reports/{sample_id}")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "App Store Review Report" in response.text


async def test_get_report_not_found_returns_404(client):
    response = await client.get(f"/api/v1/reports/{uuid.uuid4()}")

    assert response.status_code == 404
