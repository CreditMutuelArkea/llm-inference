import time
import logging

import torch
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
            
    except HTTPException:
        metrics.REQUEST_FAILURE.inc()
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        metrics.REQUEST_FAILURE.inc()
        raise HTTPException(status_code=500, detail=f"Unexpected error occurred: {e}")

    else:
        metrics.REQUEST_SUCCESS.inc()
    finally:
        torch.cuda.empty_cache()

    return ScoringResponse(
        response=[[ClassificationItem(**cat) for cat in output] for output in outputs]
    )
