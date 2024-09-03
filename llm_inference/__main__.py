import argparse

import uvicorn
from fastapi import FastAPI
from prometheus_client import make_asgi_app

from llm_inference.routes import embedding, scoring, healthcheck, guardrail
from llm_inference.model import load_pipeline, Task


app = FastAPI()
app.include_router(healthcheck.router)

# Add prometheus asgi middleware to route /metrics requests
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--model", type=str, default="cmarkea/bloomz-560m-retriever")
    parser.add_argument("--task", type=str, default="EMBEDDING")
    parser.add_argument("--dtype", type=str, default="auto")
    args = parser.parse_args()

    model_task = Task(args.task)

    if model_task == Task.EMBEDDING:
        app.include_router(embedding.router)
    elif model_task == Task.SCORING:
        app.include_router(scoring.router)
    elif model_task == Task.GUARDRAIL:
        app.include_router(guardrail.router)

    load_pipeline(model=args.model, model_task=model_task, torch_dtype=args.dtype)

    uvicorn.run(app, host=args.host, port=args.port)
