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


def load_pipeline(model: str, model_task: Task, memory_fraction: float = 0.5, **kwargs):
    """
    Load a machine learning pipeline for the specified task using the given model.

    Parameters
    ----------
    model : str
        The name or path of the model to be loaded.

    model_task : Task
        The task for which the model is to be used. Should be one of the predefined
        Task enumerations (e.g., EMBEDDING, SCORING, GUARDRAIL).

    memory_fraction : float, optional
        Fraction of GPU memory to allocate for the process (default is 0.5).

    **kwargs : keyword arguments
        Additional keyword arguments to be passed to the transformers.pipeline.

    Raises
    ------
    NotImplementedError
        If the specified task is not supported by the pipeline.

    Notes
    -----
    This function sets the device to 'cuda' if a GPU is available.
    Depending on the task, it initializes the corresponding pipeline.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(memory_fraction)

    server_pipeline = ServerPipeline()

    if model_task == Task.EMBEDDING:
        server_pipeline.pipeline = transformers.pipeline(
            "feature-extraction",
            model=model,
            device=device,
            token=os.environ["HUGGING_FACE_HUB_TOKEN"],
            **kwargs
        )
    elif model_task in [Task.SCORING, Task.GUARDRAIL]:
        server_pipeline.pipeline = transformers.pipeline(
            "text-classification",
            model=model,
            device=device,
            token=os.environ["HUGGING_FACE_HUB_TOKEN"],
            **kwargs
        )
    else:
        raise NotImplementedError("This task is not actually supported")
