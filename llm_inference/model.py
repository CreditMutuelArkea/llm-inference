import logging

import os
from enum import Enum

import torch
import transformers
from transformers import Pipeline

logger = logging.getLogger(__name__)


def singleton(class_):
    instances = {}

    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]

    return getinstance


@singleton
class ServerPipeline:
    pipeline: Pipeline


class Task(Enum):
    EMBEDDING = "EMBEDDING"
    SCORING = "SCORING"
    GUARDRAIL = "GUARDRAIL"


def load_pipeline(model: str, model_task: Task, **kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    server_pipeline = ServerPipeline()

    if model_task == Task.EMBEDDING:
        server_pipeline.pipeline = transformers.pipeline(
            "feature-extraction",
            model=model,
            device=device,
            token=os.environ["HUGGING_FACE_HUB_TOKEN"],
            **kwargs
        )
    elif model_task == Task.SCORING or model_task == Task.GUARDRAIL:
        server_pipeline.pipeline = transformers.pipeline(
            "text-classification",
            model=model,
            device=device,
            token=os.environ["HUGGING_FACE_HUB_TOKEN"],
            **kwargs
        )
    else:
        raise NotImplementedError("This task is not actually supported")
