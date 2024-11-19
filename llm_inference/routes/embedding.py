import logging
import logging

import numpy as np
from fastapi import APIRouter, status, Response, HTTPException

from llm_inference import metrics
from llm_inference.model import ServerPipeline
from llm_inference.routes.models import EmbeddingResponse, EmbeddingRequest

router = APIRouter(tags=["Embedding"])
logger = logging.getLogger(__name__)


@router.post(
    "/embed",
    summary="Vectorize the given text",
    response_description="Return a 200 (OK) HTTP status code with the resulting embedding.",
    status_code=status.HTTP_200_OK,
    response_model=EmbeddingResponse,
)
@metrics.REQUEST_TIME.time()
def inference(request: EmbeddingRequest):
    metrics.BATCH_SIZE.observe(len(request.text))

    try:
        with metrics.BATCH_INFERENCE_TIME.time():
            outputs = ServerPipeline().pipeline(request.text)

        for i in range(len(outputs)):
            if request.pooling == "mean":
                outputs[i] = np.mean(outputs[i][0], axis=0).tolist()
            elif request.pooling == "last":
                outputs[i] = outputs[i][0][-1]
            else:
                return Response("Unsupported pooling method.", status_code=400)

    except Exception as e:
        # Log des erreurs spécifiques à l'application
        logger.error(f"HTTPException: {e.detail}")
        metrics.REQUEST_FAILURE.inc()
        raise HTTPException(status_code=500, detail=f"Unexpected error occurred: {e.detail}")
    
    else:
        metrics.REQUEST_SUCCESS.inc()
    return EmbeddingResponse(embedding=outputs)