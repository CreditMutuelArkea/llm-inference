from prometheus_client import Summary, Counter


REQUEST_TIME = Summary("request_seconds", "Time spent in the request")
BATCH_INFERENCE_TIME = Summary(
    "batch_inference_seconds", "Time spent during inference of the batch."
)
REQUEST_SUCCESS = Counter("request_success", "Number of successful requests.")
REQUEST_FAILURE = Counter("request_fail", "Number of failed requests.")
BATCH_SIZE = Summary("request_batch_size", "The batch size of the request.")
