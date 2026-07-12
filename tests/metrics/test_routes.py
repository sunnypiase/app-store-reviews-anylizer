import uuid


async def test_get_metrics_for_existing_sample(client, make_sample):
    sample_id = await make_sample(
        [
            {"rating": 5, "title": "Great", "content": "Love it"},
            {"rating": 5, "title": "Great", "content": "Love it"},
            {"rating": 1, "title": "Bad", "content": "Hate it"},
            {"rating": 3, "title": "Ok", "content": "It's fine"},
        ]
    )

    response = await client.get(f"/api/v1/metrics/{sample_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["review_count"] == 4
    assert body["average_rating"] == 3.5
    assert body["rating_distribution"]["5"]["count"] == 2


async def test_get_metrics_not_found_returns_404(client):
    response = await client.get(f"/api/v1/metrics/{uuid.uuid4()}")

    assert response.status_code == 404
