# BUILD
ARG PYTHON_VERSION="3.10"
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04 AS runtime

USER root
STOPSIGNAL SIGINT

ENV PROJECT="llm_inference"
ENV PACKAGE="llm_inference"

RUN apt-get update && \
    apt-get install -y python3-pip python3-dev python3-venv && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /usr/app
RUN pip install virtualenv && python3 -m venv /usr/app/venv
ENV PATH="/usr/app/venv/bin:$PATH"

COPY setup.py ./requirements.txt ./requirements-dev.txt ./index.md ./MANIFEST.in ./
COPY ./${PACKAGE} /usr/app/${PACKAGE}

RUN pip install --upgrade pip && pip install .

ENV DKR_USER "python"
ENV GUNICORN_LISTEN_ADDRESS="0.0.0.0"
ENV GUNICORN_LISTEN_PORT=8080
ENV GUNICORN_TIMEOUT=120
ENV IN_DOCKER_CONTAINER Yes
ENV HF_MODEL="cmarkea/bloomz-3b-nli"

RUN groupadd -g 1003 "$DKR_USER" && useradd -r -u 1003 -g "$DKR_USER" "$DKR_USER"
RUN chown "$DKR_USER":"$DKR_USER" /usr/app

USER $DKR_USER
ENV PATH="/usr/app/venv/bin:$PATH"
EXPOSE ${GUNICORN_LISTEN_PORT}

ENTRYPOINT ["/usr/app/venv/bin/python", "-m", "llm_inference"]
CMD ["--host", "${GUNICORN_LISTEN_ADDRESS}", "--port", "${GUNICORN_LISTEN_PORT}", "--model", "${HF_MODEL}"]
