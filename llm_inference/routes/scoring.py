import time
import logging

from fastapi import APIRouter, HTTPException, status

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
                [
                    {"text": context.context, "text_pair": context.query}
                    for context in request.contexts
                ],
                function_to_apply="softmax",
                top_k=None,
            )
            
    except Exception as e:
        # Log des erreurs spécifiques à l'application
        logger.error(f"HTTPException: {e.detail}")
        metrics.REQUEST_FAILURE.inc()
        raise HTTPException(status_code=500, detail=f"Unexpected error occurred: {e.detail}")
        
    else:
        metrics.REQUEST_SUCCESS.inc()

    return ScoringResponse(
        response=[[ClassificationItem(**cat) for cat in output] for output in outputs]
    )
