import uuid


async def test_get_insight_for_existing_sample(client, make_sample):
    sample_id = await make_sample(
        [
            {
                "rating": 1,
                "title": "Billing issue",
                "content": "Customer service never responds, charged twice",
            },
            {
                "rating": 5,
                "title": "Great",
                "content": "This app is fantastic, I use it every day",
            },
        ]
    )

    response = await client.get(f"/api/v1/insights/{sample_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["review_count"] == 2
    distribution = body["sentiment_distribution"]
    assert distribution["positive"] + distribution["neutral"] + distribution["negative"] == 2


async def test_get_insight_not_found_returns_404(client):
    response = await client.get(f"/api/v1/insights/{uuid.uuid4()}")

    assert response.status_code == 404
