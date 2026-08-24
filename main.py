from __future__ import annotations

import logging
import os
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from classifier import  LLMClassifier, Classifier
from service import classify_plaid_response
from taxonomy import DEFAULT_TAXONOMY


load_dotenv()

logger = logging.getLogger("transaction_classifier")

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


def get_classifier() -> Classifier:
    """Prefer the LLM backend when a key is configured; fall back to the
    deterministic mock backend on any init failure so the service never
    fails to start (and demos/tests still work without an API key)."""
    print(os.environ.get("GEMINI_API_KEY"))
    # if os.environ.get("GEMINI_API_KEY"):
    #     try:
    #         return LLMClassifier()
    #     except Exception as exc:  # noqa: BLE001
    #         logger.warning("Falling back to MockClassifier: %s", exc)
    #         return MockClassifier()
    return LLMClassifier()


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


@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Transaction Classifier Demo</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-gray-50 text-gray-900 font-sans">
        <div class="max-w-7xl mx-auto px-4 py-8">
            <header class="mb-8 flex justify-between items-center">
                <div>
                    <h1 class="text-3xl font-extrabold text-gray-900 tracking-tight">Transaction Classification Dashboard</h1>
                    <p class="mt-1 text-sm text-gray-500">Paste your raw Plaid API response JSON below.</p>
                </div>
                <div>
                    <button onclick="runClassification()" id="run-btn" class="bg-indigo-600 hover:bg-indigo-700 text-white font-medium px-5 py-2.5 rounded-lg shadow transition">
                        Run Classification
                    </button>
                </div>
            </header>

            <div class="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
                <div class="lg:col-span-2 bg-white p-5 rounded-xl border border-gray-200 shadow-sm">
                    <label class="block text-sm font-medium text-gray-700 mb-2">Plaid JSON Payload</label>
                    <textarea id="json-input" rows="10" class="w-full font-mono text-xs p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:outline-none" placeholder="Paste raw Plaid response JSON here..."></textarea>
                </div>
                <div class="bg-white p-5 rounded-xl border border-gray-200 shadow-sm flex flex-col justify-between">
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-2">Upload JSON File</label>
                        <input type="file" id="file-input" accept=".json" onchange="handleFileUpload(event)" class="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-indigo-50 file:text-indigo-700 hover:file:bg-indigo-100 cursor-pointer"/>
                    </div>
                    <div id="loading" class="hidden text-indigo-600 font-medium animate-pulse mt-4">
                        Processing transactions...
                    </div>
                </div>
            </div>

            <div id="summary" class="hidden grid grid-cols-2 md:grid-cols-5 gap-4 mb-6 text-sm"></div>

            <div class="bg-white shadow-sm rounded-xl border border-gray-200 overflow-hidden">
                <table class="min-w-full divide-y divide-gray-200">
                    <thead class="bg-gray-50">
                        <tr>
                            <th class="px-6 py-3 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider">Transaction ID</th>
                            <th class="px-6 py-3 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider">Predicted Class</th>
                            <th class="px-6 py-3 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider">Type</th>
                            <th class="px-6 py-3 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider">Direction</th>
                            <th class="px-6 py-3 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider">Confidence</th>
                            <th class="px-6 py-3 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider">Reason Code</th>
                            <th class="px-6 py-3 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider">Status</th>
                        </tr>
                    </thead>
                    <tbody id="results-body" class="bg-white divide-y divide-gray-200 text-sm">
                        <tr>
                            <td colspan="7" class="px-6 py-8 text-center text-gray-400">Paste your Plaid JSON response above and click "Run Classification".</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>

        <script>
            function handleFileUpload(event) {
                const file = event.target.files[0];
                if (!file) return;
                const reader = new FileReader();
                reader.onload = function(e) {
                    try {
                        const parsed = JSON.parse(e.target.result);
                        document.getElementById('json-input').value = JSON.stringify(parsed, null, 2);
                    } catch (err) {
                        alert('Invalid JSON file format.');
                    }
                };
                reader.readAsText(file);
            }

            function statBox(label, value) {
                return `<div class="bg-white p-4 rounded-xl border border-gray-200"><div class="text-xs text-gray-500 uppercase tracking-wide">${label}</div><div class="text-lg font-bold text-gray-900 mt-1">${value}</div></div>`;
            }

            async function runClassification() {
                const btn = document.getElementById('run-btn');
                const loading = document.getElementById('loading');
                const tbody = document.getElementById('results-body');
                const summaryEl = document.getElementById('summary');
                const rawInput = document.getElementById('json-input').value.trim();

                if (!rawInput) {
                    alert('Please provide Plaid JSON data.');
                    return;
                }

                let transactions;
                try {
                    transactions = JSON.parse(rawInput);
                } catch (err) {
                    alert('JSON Parse Error: ' + err.message);
                    return;
                }

                btn.disabled = true;
                loading.classList.remove('hidden');

                try {
                    const res = await fetch('/api/classify', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ transactions })
                    });

                    const data = await res.json();
                    if (!res.ok) throw new Error(data.detail || 'Server error');

                    const s = data.summary;
                    summaryEl.innerHTML = [
                        statBox('Total', s.total_transactions),
                        statBox('Income', '$' + s.income_total.toFixed(2)),
                        statBox('Expense', '$' + s.expense_total.toFixed(2)),
                        statBox('Needs Review', s.needs_review_count),
                        statBox('Guessed', s.guessed_count),
                    ].join('');
                    summaryEl.classList.remove('hidden');

                    tbody.innerHTML = '';

                    data.transactions.forEach(t => {
                        const c = t.classification;
                        const tr = document.createElement('tr');
                        const statusBadge = c.needs_review
                            ? '<span class="px-2.5 py-1 text-xs font-medium rounded-full bg-amber-100 text-amber-800">Needs Review</span>'
                            : '<span class="px-2.5 py-1 text-xs font-medium rounded-full bg-emerald-100 text-emerald-800">Passed</span>';
                        const guessBadge = c.is_guess ? ' <span class="text-xs text-gray-400">(guess)</span>' : '';
                        const txnId = t.transaction_id || t.id || '';

                        tr.innerHTML = `
                            <td class="px-6 py-4 whitespace-nowrap font-mono text-gray-500 text-xs">${txnId}</td>
                            <td class="px-6 py-4 whitespace-nowrap font-medium text-gray-900">${c.predicted_class}${guessBadge}</td>
                            <td class="px-6 py-4 whitespace-nowrap text-gray-600 capitalize">${c.transaction_type}</td>
                            <td class="px-6 py-4 whitespace-nowrap text-gray-600 capitalize">${c.direction}</td>
                            <td class="px-6 py-4 whitespace-nowrap text-gray-600">${(c.confidence * 100).toFixed(0)}%</td>
                            <td class="px-6 py-4 whitespace-nowrap text-xs font-mono text-gray-500">${c.reason_code}</td>
                            <td class="px-6 py-4 whitespace-nowrap">${statusBadge}</td>
                        `;
                        tbody.appendChild(tr);
                    });
                } catch (err) {
                    alert('Error running classification: ' + err.message);
                } finally {
                    btn.disabled = false;
                    loading.classList.add('hidden');
                }
            }
        </script>
    </body>
    </html>
    """
