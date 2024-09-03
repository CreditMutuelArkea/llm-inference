import time
import logging

from fastapi import APIRouter, status

from llm_inference import metrics
from llm_inference.routes.models import (
    ScoringRequest,
    ScoringResponse,
    ClassificationItem,
)
from llm_inference.model import ServerPipeline

router = APIRouter(tags=["Scoring"])
logger = logging.getLogger(__name__)


@router.post(
    "/score",
    summary="Evaluate multiple embeddings with respect of the query asked",
    response_description="Return a 200 (OK) HTTP status code.",
    status_code=status.HTTP_200_OK,
    response_model=ScoringResponse,
)
@metrics.REQUEST_TIME.time()
def inference(request: ScoringRequest) -> ScoringResponse:
    metrics.BATCH_SIZE.observe(len(request.contexts))

    try:
        with metrics.BATCH_INFERENCE_TIME.time():
            outputs = ServerPipeline().pipeline(
                [{"text": context.context, "text_pair": context.query} for context in request.contexts],
                function_to_apply="softmax",
                return_all_scores=True,
            )
    except Exception as e:
        metrics.REQUEST_FAILURE.inc()
        raise e
    else:
        metrics.REQUEST_SUCCESS.inc()

    return ScoringResponse(response=[[ClassificationItem(**cat) for cat in output] for output in outputs])
