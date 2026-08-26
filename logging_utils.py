"""
Structured logging: one JSON object per log line, tagged with a per-request
correlation id, so a single classification call's full trail -- which flow
classified each transaction, where fallbacks/overrides triggered, what the
batch-level diagnostics were -- can be isolated from Render's interleaved
log stream by filtering on one request_id, instead of grepping strings.

Usage:
    from logging_utils import configure_logging, log_event, ensure_request_id

    configure_logging()                 # once, at process start (main.py)
    logger = logging.getLogger(__name__)

    ensure_request_id()                 # once per logical request/call
    log_event(logger, "normalization_complete", normalized_count=12, skipped_count=0)
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from typing import Any

_REQUEST_ID: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

# LogRecord attribute names to exclude when promoting `extra=` fields to
# top-level JSON keys, so we don't clobber/duplicate logging internals.
_RESERVED_LOG_RECORD_ATTRS = set(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()
) | {"message", "asctime"}


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]


def set_request_id(request_id: str) -> None:
    _REQUEST_ID.set(request_id)


def get_request_id() -> str:
    return _REQUEST_ID.get()


def ensure_request_id() -> str:
    """Returns the current request id, generating and setting a new one if
    none is set yet. The FastAPI middleware sets one per HTTP request; this
    covers callers that invoke the service layer directly (CLI, worker,
    tests) so every log line still has *some* correlation id."""
    current = _REQUEST_ID.get()
    if current != "-":
        return current
    rid = new_request_id()
    _REQUEST_ID.set(rid)
    return rid


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _REQUEST_ID.get()
        return True


class JsonFormatter(logging.Formatter):
    """Renders each log record as one JSON object per line. Fields passed
    via `extra=` in a logging call become top-level JSON keys (rather than
    being buried in a formatted message string), so they're filterable as
    structured fields in Render's log viewer or any JSON-aware aggregator
    (e.g. flow="llm_override_regular_w2")."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "request_id": getattr(record, "request_id", "-"),
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in _RESERVED_LOG_RECORD_ATTRS or key == "request_id":
                continue
            try:
                json.dumps(value)
                payload[key] = value
            except TypeError:
                payload[key] = repr(value)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def configure_logging(level: str | None = None) -> None:
    """Call once at process start. Safe to call more than once -- replaces
    handlers rather than stacking them (relevant if a test harness or
    reloader re-imports main.py)."""
    root = logging.getLogger()
    root.setLevel(level or os.environ.get("LOG_LEVEL", "INFO"))
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    handler.addFilter(_RequestIdFilter())
    root.addHandler(handler)


def log_event(
    logger: logging.Logger,
    event: str,
    level: int = logging.INFO,
    exc_info: bool = False,
    **fields: Any,
) -> None:
    """Structured log helper. `event` is a short, stable, snake_case name
    meant to be grepped/dashboarded on (e.g. "regular_w2_override_applied"),
    distinct from free-text messages. Any keyword args become top-level
    JSON fields on that log line."""
    logger.log(level, event, extra={"event": event, **fields}, exc_info=exc_info)