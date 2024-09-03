import os
from unittest import mock
import torch

import pytest
from llm_inference.model import load_pipeline

from llm_inference.model import Task


@pytest.mark.parametrize("model_task,pipeline_type", [
    (Task.SCORING, "text-classification"),
    (Task.GUARDRAIL, "text-classification"),
    (Task.EMBEDDING, "feature-extraction")
])
@mock.patch("llm_inference.model.transformers.pipeline")
@mock.patch.dict(os.environ, {"HUGGING_FACE_HUB_TOKEN": "token"})
def test_load_pipeline_should_succeed(pipeline, model_task, pipeline_type):
    # Given
    # When
    load_pipeline(model="model", model_task=model_task)
    # Then
    pipeline.assert_called_once_with(pipeline_type, model="model", device=torch.device(type='cpu'), token="token")

