FROM python:3.11.9-slim-bookworm AS app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 nawei \
    && useradd --uid 10001 --gid nawei --create-home --shell /usr/sbin/nologin nawei

COPY deploy/requirements-app.txt /tmp/requirements-app.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install --requirement /tmp/requirements-app.txt

WORKDIR /app
COPY --chown=nawei:nawei app_protocols /app/app_protocols
COPY --chown=nawei:nawei services/__init__.py /app/services/__init__.py
COPY --chown=nawei:nawei services/gateway /app/services/gateway
COPY --chown=nawei:nawei services/llm /app/services/llm
COPY --chown=nawei:nawei services/knowledge /app/services/knowledge
COPY --chown=nawei:nawei deploy/scripts /app/deploy/scripts
COPY --chown=nawei:nawei deploy/migrations /app/deploy/migrations

RUN mkdir -p /models/huggingface /models/sentence-transformers /var/lib/nawei \
    && chown -R nawei:nawei /models /var/lib/nawei

USER nawei

CMD ["python", "-m", "services.gateway.run"]

FROM app AS knowledge
USER root
COPY deploy/requirements-embeddings.txt /tmp/requirements-embeddings.txt
RUN python -m pip install --requirement /tmp/requirements-embeddings.txt
USER nawei
CMD ["python", "-m", "services.knowledge.run"]
