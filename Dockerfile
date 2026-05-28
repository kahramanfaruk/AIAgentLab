# Single image used for both the API and the UI services; the service command
# selects which process to run (see docker-compose.yml).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first for better layer caching.
COPY pyproject.toml README.md ./
COPY agent ./agent
COPY api ./api
COPY config ./config
COPY ui ./ui

RUN pip install --no-cache-dir -e .

EXPOSE 8000 8501

# Default to the API; the UI service overrides this command.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
