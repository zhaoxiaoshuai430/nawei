FROM python:3.11.9-slim-bookworm

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

RUN mkdir -p /models/huggingface /models/sentence-transformers \
    && chown -R nawei:nawei /models

USER nawei

CMD ["python", "-m", "services.gateway.run"]
