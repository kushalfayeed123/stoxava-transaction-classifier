from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from classifier import LLMClassifier, Classifier, MockClassifier
from llm_provider import ProviderConfig
from service import classify_plaid_response
from taxonomy import DEFAULT_TAXONOMY


load_dotenv()

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
# Root logging config so `logger.info(...)` calls actually surface somewhere
# (uvicorn's default config otherwise silences app-level loggers depending
# on how the process is started).
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

logger = logging.getLogger("transaction_classifier")
access_logger = logging.getLogger("transaction_classifier.access")

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="Stoxava Transaction Classifier",
    description=(
        "Middleware service: accepts raw Plaid transaction payloads from a "
        "backend, normalizes across account/schema variations, classifies "
        "each transaction against a taxonomy, and returns the complete "
        "transaction plus a `classification` object the backend can filter "
        "on (predicted_class, confidence, direction, transaction_type, "
        "needs_review, is_guess).\n\n"
        "Try `POST /api/classify` below with **Try it out** -- it comes "
        "pre-filled with a real Plaid `/transactions/sync` sample payload."
    ),
    version="0.2.0",
    contact={"name": "Stoxava Engineering"},
    openapi_tags=[
        {"name": "classification", "description": "Classify transactions and get them back enriched."},
        {"name": "ops", "description": "Health/status endpoints for uptime checks and load balancers."},
    ],
)


# --------------------------------------------------------------------------
# Request logging middleware
# --------------------------------------------------------------------------
# Logs method, path, client IP, and User-Agent for every request, plus
# status + latency once the response is ready. This is what tells you
# *who* is hitting /health repeatedly: a platform health checker, an
# uptime pinger, multiple worker processes, or a scanner will each show
# a distinct UA/IP/cadence pattern in these lines.
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "-")

    response = await call_next(request)

    duration_ms = (time.perf_counter() - start) * 1000
    # Health checks are extremely high-volume and low-signal once you've
    # diagnosed the source; keep them at DEBUG so normal INFO-level logs
    # aren't drowned out, but they're still available if you bump the level.
    log_fn = access_logger.debug if request.url.path == "/health" else access_logger.info
    log_fn(
        "%s %s -> %s (%.1fms) ip=%s ua=%s",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
        client_ip,
        user_agent,
    )
    return response


def get_classifier():
    try:
        return LLMClassifier(provider_config=ProviderConfig())
    except (ImportError, RuntimeError, ValueError) as exc:
        logging.getLogger(__name__).warning(
            "Falling back to MockClassifier: %s", exc
        )
        return MockClassifier()


classifier: Classifier = get_classifier()


# --------------------------------------------------------------------------
# Request/response contracts
# --------------------------------------------------------------------------

_EXAMPLE_PLAID_SYNC_PAYLOAD = {
    "transactions_update_status": "HISTORICAL_UPDATE_COMPLETE",
    "accounts": [
        {
            "account_id": "ZNW9DqgoZVSjMmLMP3zKiNkewgX1kguADlqE7",
            "balances": {"available": 82098.22, "current": 82098.22, "iso_currency_code": "USD"},
            "mask": "1384",
            "name": "Checking",
            "type": "depository",
            "subtype": "checking",
        }
    ],
    "added": [
        {
            "account_id": "ZNW9DqgoZVSjMmLMP3zKiNkewgX1kguADlqE7",
            "amount": 105.07,
            "date": "2026-07-16",
            "merchant_name": "Costco",
            "name": "Costco",
            "category": ["Shops", "Warehouses and Wholesale Stores"],
            "personal_finance_category": {"detailed": "GENERAL_MERCHANDISE_SUPERSTORES", "primary": "GENERAL_MERCHANDISE"},
            "transaction_id": "qBy7pRXzGVIkzKlz6QenfAZ6bnyDEGUAl5oxo",
        },
        {
            "account_id": "ZNW9DqgoZVSjMmLMP3zKiNkewgX1kguADlqE7",
            "amount": 154.53,
            "date": "2026-07-16",
            "merchant_name": "Smart & Final",
            "name": "POS SMART AND FINAL 111",
            "category": ["Shops", "Supermarkets and Groceries"],
            "personal_finance_category": {"detailed": "FOOD_AND_DRINK_GROCERIES", "primary": "FOOD_AND_DRINK"},
            "transaction_id": "KEDM9naxKQTJG39GWPa5tm3koQvL6eSWojnJK",
        },
        {
            "account_id": "ZNW9DqgoZVSjMmLMP3zKiNkewgX1kguADlqE7",
            "amount": 31.19,
            "date": "2026-07-16",
            "merchant_name": "Vidalia's Restaurant",
            "name": "VIDALIA'S RESTAURANT",
            "category": ["Food and Drink", "Restaurants"],
            "personal_finance_category": {"detailed": "FOOD_AND_DRINK_RESTAURANT", "primary": "FOOD_AND_DRINK"},
            "transaction_id": "r3neyJRBGVTvMrWMmlQ3F6nzxZLpQ9CqbXomG",
        },
    ],
    "modified": [],
    "removed": [],
    "has_more": True,
}


class ClassificationRequest(BaseModel):
    # Accepts either the full Plaid /transactions/sync-style envelope
    # ({"accounts": [...], "added": [...]}), a {"transactions": [...]}
    # wrapper, or a bare list of transaction objects.
    transactions: Any = Field(..., description="Raw Plaid payload or list of transactions.")
    sign_convention: str = Field(
        "standard",
        description=(
            "'standard' uses Plaid's own sign convention (positive = money "
            "out, negative = money in). 'flipped' inverts it for upstream "
            "sources that report the opposite way."
        ),
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"transactions": _EXAMPLE_PLAID_SYNC_PAYLOAD, "sign_convention": "standard"}
            ]
        }
    }


class Classification(BaseModel):
    predicted_class: str
    confidence: float
    alternative_class: Optional[str]
    reason_code: str
    needs_review: bool
    is_guess: bool
    direction: str          # "debit" | "credit"
    transaction_type: str   # "income" | "expense" | "transfer"


class ClassificationSummary(BaseModel):
    total_transactions: int
    skipped_transactions: int
    income_total: float
    expense_total: float
    needs_review_count: int
    guessed_count: int


class ClassificationResponse(BaseModel):
    backend: str
    summary: ClassificationSummary
    transactions: list[dict[str, Any]]  # original transaction fields + "classification"
    skipped: list[dict[str, Any]]


@app.get("/health", tags=["ops"], summary="Liveness/readiness probe")
def health() -> dict[str, str]:
    return {"status": "ok", "backend": type(classifier).__name__}


@app.post(
    "/api/classify",
    response_model=ClassificationResponse,
    tags=["classification"],
    summary="Classify a batch of Plaid transactions",
    response_description="Original transactions enriched with a `classification` object, plus a batch summary.",
)
def classify_transactions(payload: ClassificationRequest) -> dict[str, Any]:
    raw_data = payload.transactions

    if not raw_data:
        raise HTTPException(status_code=400, detail="Payload is empty.")

    try:
        result = classify_plaid_response(
            raw_data,
            classifier,
            taxonomy=DEFAULT_TAXONOMY,
            sign_convention=payload.sign_convention,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Classification failed")
        raise HTTPException(status_code=500, detail=f"Classification failed: {exc}") from exc

    if result["summary"]["total_transactions"] == 0:
        raise HTTPException(status_code=400, detail="No valid transactions found in Plaid payload.")

    return result


# Read once at import time instead of re-rendering a giant Python string
# literal (and re-allocating it) on every single GET / request.
_INDEX_HTML = (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _INDEX_HTML