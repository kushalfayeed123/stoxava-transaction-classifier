# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

# uv gives fast, reproducible installs from uv.lock (already committed).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Install deps first (cache-friendly layer) before copying source.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . .
RUN uv sync --frozen --no-dev

# Change ownership to non-root user
RUN chown -R appuser:appuser /app
USER appuser

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

# Health check for container orchestration
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)" || exit 1

# $PORT is set by most free hosts (Render, Railway, Fly); default to 8000
# for plain `docker run`.
# Use gunicorn with workers for production (better than uvicorn single process)
CMD ["gunicorn", "main:app", "--workers", "2", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:${PORT:-8000}", "--timeout", "60", "--access-logfile", "-", "--error-logfile", "-"]
