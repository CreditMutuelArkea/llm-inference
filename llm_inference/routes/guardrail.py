import time
import logging

from fastapi import APIRouter, status

from llm_inference import metrics
from llm_inference.routes.models import (
    GuardrailRequest,
    GuardrailResponse,
    ClassificationItem,
)
from llm_inference.model import ServerPipeline

router = APIRouter(tags=["Guardrail"])
logger = logging.getLogger(__name__)


@router.post(
    "/guardrail",
    summary="Evaluate a sentence and detect the toxicity of a text.",
    response_description="Return a 200 (OK) HTTP status code.",
    status_code=status.HTTP_200_OK,
    response_model=GuardrailResponse,
)
@metrics.REQUEST_TIME.time()
def inference(request: GuardrailRequest) -> GuardrailResponse:
    logger.info(f"Input request with size : {len(request.text)}")

    try:
        with metrics.BATCH_INFERENCE_TIME.time():
            outputs = ServerPipeline().pipeline(
                request.text, function_to_apply="sigmoid", top_k=None
            )
    except Exception as e:
        metrics.REQUEST_FAILURE.inc()
        raise e
    else:
        metrics.REQUEST_SUCCESS.inc()

    return GuardrailResponse(
        response=[[ClassificationItem(**cat) for cat in output] for output in outputs]
    )
