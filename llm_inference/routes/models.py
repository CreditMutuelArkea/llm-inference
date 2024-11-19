import logging
from typing import List

from pydantic import BaseModel, StringConstraints
from typing import Annotated


logger = logging.getLogger(__name__)


class ScoringItem(BaseModel):
    query: Annotated[str, StringConstraints(max_length=3000)]
    context: Annotated[str, StringConstraints(max_length=3000)]

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
