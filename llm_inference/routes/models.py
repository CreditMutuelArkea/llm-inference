import logging
from typing import List

from pydantic import BaseModel


logger = logging.getLogger(__name__)


class ScoringItem(BaseModel):
    query: str
    context: str


class ClassificationItem(BaseModel):
    label: str
    score: float


class ScoringRequest(BaseModel):
    contexts: List[ScoringItem]


class EmbeddingRequest(BaseModel):
    text: List[str]
    pooling: str


class GuardrailRequest(BaseModel):
    text: List[str]


class EmbeddingResponse(BaseModel):
    embedding: List[List[float]]


class ScoringResponse(BaseModel):
    response: List[List[ClassificationItem]]


class GuardrailResponse(BaseModel):
    response: List[List[ClassificationItem]]
