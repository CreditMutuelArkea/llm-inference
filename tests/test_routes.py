import pytest
import random

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
    return TestClient(app)


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
    

def test_scoring_unprocessable_entity_error(client):
    def generate_random_sentence(length):
        words = [
            "lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing", "elit", "sed", 
            "do", "eiusmod", "tempor", "incididunt", "ut", "labore", "et", "dolore", "magna", 
            "aliqua", "ut", "enim", "ad", "minim", "veniam", "quis", "nostrud", "exercitation", 
            "ullamco", "laboris", "nisi", "ut", "aliquip", "ex", "ea", "commodo", "consequat",
            "duis", "aute", "irure", "dolor", "in", "reprehenderit", "in", "voluptate", "velit",
            "esse", "cillum", "dolore", "eu", "fugiat", "nulla", "pariatur", "excepteur", "sint",
            "occaecat", "cupidatat", "non", "proident", "sunt", "in", "culpa", "qui", "officia",
            "deserunt", "mollit", "anim", "id", "est", "laborum",
        ]
        
        sentence = []
        current_length = 0
        
        while current_length < length:
            word = random.choice(words) 
            sentence.append(word)
            current_length += len(word) + 1 # longueur du mot et l'espace suivant

        return ' '.join(sentence).strip()

    random_sentence = generate_random_sentence(length=3001)
    data_with_invalid_context = {
        "contexts": [{"query": "", "context": f"{random_sentence}"}]
    }
    response = client.post("/score", json=data_with_invalid_context)
    assert response.status_code == 422 

    data_with_invalid_query = {
        "contexts": [{"query": f"{random_sentence}", "context": ""}]
    }
    response = client.post("/score", json=data_with_invalid_query)
    assert response.status_code == 422 

