import pytest

import unittest.mock as mock
from fastapi.testclient import TestClient

from llm_inference.routes.models import (
    EmbeddingRequest,
    ScoringRequest,
    ScoringItem,
    GuardrailRequest,
)

from llm_inference.__main__ import app
from llm_inference.routes import embedding, scoring, guardrail


@pytest.fixture
def client():
    app.include_router(embedding.router)
    app.include_router(scoring.router)
    app.include_router(guardrail.router)
    client = TestClient(app)
    return client


def test_ping_should_succeed(client):
    # Given
    # When
    response = client.get("/ping")
    # Then
    assert response.status_code == 200


@pytest.mark.parametrize(
    "pooling,expected_status_code",
    [("mean", 200), ("last", 200), ("not_a_valid_pooling_value", 400)],
)
@mock.patch("llm_inference.routes.embedding.ServerPipeline")
def test_embedding_should_succeed(pipeline, client, pooling, expected_status_code):
    # Given

    pipeline.return_value.pipeline.return_value = [
        [
            [
                [0.45442174673080443],
                [-0.09678886085748672],
                [0.10536841936409473],
            ]
        ]
    ]
    embedding_request = EmbeddingRequest(text=["one", "two", "three"], pooling=pooling)
    # When
    response = client.post("/embed", json=embedding_request.model_dump())
    # Then
    assert response.status_code == expected_status_code


@mock.patch("llm_inference.routes.scoring.ServerPipeline")
def test_scoring_should_succeed(pipeline, client):
    # Given

    pipeline.return_value.pipeline.return_value = [
        [{"label": "POSITIVE", "score": 0.6970456838607788}]
    ]
    scoring_request = ScoringRequest(
        contexts=[ScoringItem(query="", context="Lorem ipsum")]
    )
    # When
    response = client.post("/score", json=scoring_request.model_dump())
    # Then
    assert response.status_code == 200


@mock.patch("llm_inference.routes.guardrail.ServerPipeline")
def test_guardrail_should_succeed(pipeline, client):
    # Given

    pipeline.return_value.pipeline.return_value = [
        [{"label": "POSITIVE", "score": 0.6970456838607788}]
    ]
    embedding_request = GuardrailRequest(text=["one", "two", "three"])
    # When
    response = client.post("/guardrail", json=embedding_request.model_dump())
    # Then
    assert response.status_code == 200
