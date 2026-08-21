# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

# uv gives fast, reproducible installs from uv.lock (already committed).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install deps first (cache-friendly layer) before copying source.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . .
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

EXPOSE 8000

# $PORT is set by most free hosts (Render, Railway, Fly); default to 8000
# for plain `docker run`.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
