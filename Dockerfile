FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# System deps (keep minimal; add more only when needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry (pin for reproducible builds)
RUN pip install --no-cache-dir poetry==1.8.3

# Copy dependency definitions first (for better layer caching)
COPY pyproject.toml poetry.lock README.md ./

# Copy package sources (pyproject.toml packages = { include = "qts_core", from = "qts_core/src" })
COPY qts_core/src ./qts_core/src

# Copy Hydra configuration
COPY conf ./conf

# Install dependencies and the package (no venv inside container)
RUN poetry config virtualenvs.create false \
    && poetry install --only main --no-interaction --no-ansi

EXPOSE 8501

# Default command (overridden in docker-compose)
CMD ["python", "-m", "qts_core.main_live"]
