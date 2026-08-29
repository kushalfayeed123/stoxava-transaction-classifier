# Stoxava Transaction Classifier

A middleware service that sits between your backend and raw Plaid
transaction data. It normalizes transactions from multiple accounts (even
when their schemas differ), classifies every transaction against a
taxonomy, and hands back the **complete original transaction** plus a
`classification` object your backend can filter/sort on.

## Why a middleware, not a library call

Your backend can call this over HTTP (`POST /api/classify`) from any
language/service, so the classification logic, taxonomy, LLM key, and
prompt live in one place and can be scaled, deployed, and versioned
independently of the main backend.

## Running it locally

```bash
uv sync
uv run uvicorn main:app --reload
```

- Set `LLM_PROVIDER` and `LLM_API_KEY` to use the real LLM backend. 
  **Recommended:** `LLM_PROVIDER=groq` with key from [console.groq.com](https://console.groq.com) 
  (uses `llama-3.1-8b-instant`: ~50ms latency, 30k RPM free tier).
  Other options: `LLM_PROVIDER=gemini` (Gemini 1.5 Flash-8B), `LLM_PROVIDER=nvidia` (Nemotron 3 Nano 30B), `LLM_PROVIDER=openrouter`.
- Without API keys, the service automatically falls back to `MockClassifier`,
  a deterministic rule engine, so the service never fails to start and
  always returns a result.
- `GET /health` → `{"status": "ok", "backend": "MockClassifier", "uptime_seconds": 123.4}`
- `GET /` → a small HTML dashboard for manually pasting a Plaid payload and
  inspecting results.
- `GET /docs` → **Swagger UI** (interactive, "Try it out" pre-filled with a
  real Plaid `/transactions/sync` sample). Share this URL with other devs —
  they can read every field, hit the endpoint, and see real responses
  without writing any client code.
- `GET /redoc` → a cleaner read-only spec view, if that's preferred for
  circulating the contract.
- `GET /openapi.json` → the raw spec, if devs want to import it into
  Postman/Insomnia.

## Free hosting (quick test before the VPS)

The service is a single Docker image with no external dependencies besides
an optional `LLM_API_KEY`, so any free container host works. Two ready
to go:

### Render (recommended — simplest, has a free web-service tier)

1. Push this folder to a GitHub repo.
2. On [render.com](https://render.com): **New → Blueprint**, point it at
   the repo. Render reads `render.yaml` in this project automatically and
   creates the service — no manual config needed.
3. (Optional) In the service's **Environment** tab, add:
   - `LLM_PROVIDER=groq` (fastest for classification)
   - `LLM_API_KEY=gsk_...` (get from console.groq.com)
   - `LOG_LEVEL=INFO`
4. You'll get a URL like `https://transaction-classifier-xxxx.onrender.com`
   — share `.../docs` with other devs.

Free-tier caveat: the instance sleeps after ~15 minutes idle and takes a
few seconds to cold-start on the next request — fine for testing, just hit
`/health` once before a live demo to warm it up.

### Fly.io (alternative — always-on free tier, better for demos)

```bash
brew install flyctl   # or see fly.io/docs/hands-on/install-flyctl
fly launch --no-deploy   # detects the Dockerfile, uses fly.toml in this repo
fly secrets set LLM_PROVIDER=groq LLM_API_KEY=gsk_...   # optional
fly deploy
```

Fly free tier: 3 shared-CPU VMs, 512MB RAM each, 160GB monthly transfer,
**always-on** (no sleep like Render). Great for live demos.

### Docker (any VPS)

```bash
docker build -t transaction-classifier .
docker run -p 8000:8000 \
  -e LLM_PROVIDER=groq \
  -e LLM_API_KEY=gsk_... \
  -e PORT=8000 \
  transaction-classifier
```

**Recommended VPS specs:** 1 vCPU, 1GB RAM, ~$4-6/mo (DigitalOcean, Linode, Hetzner, etc.)

## API contract

### `POST /api/classify`

Request body:

```json
{
  "transactions": { "accounts": [...], "added": [...] },
  "sign_convention": "standard"
}
```

`transactions` accepts any of:
- The full Plaid `/transactions/sync` envelope (`{"accounts": [...], "added": [...]}`).
- `{"transactions": [...]}`.
- A bare `[...]` list of transaction objects.

`sign_convention` is `"standard"` (Plaid's own convention: positive =
money OUT, negative = money IN) or `"flipped"` for upstream sources that
report the opposite way.

Response body:

```json
{
  "backend": "MockClassifier",
  "summary": {
    "total_transactions": 31,
    "skipped_transactions": 0,
    "income_total": 6050.0,
    "expense_total": 4074.66,
    "needs_review_count": 2,
    "guessed_count": 2
  },
  "transactions": [
    {
      "...every original field from the input transaction, untouched...": "...",
      "classification": {
        "predicted_class": "Groceries",
        "confidence": 0.9,
        "alternative_class": "Uncategorized",
        "reason_code": "KEYWORD_MATCH",
        "needs_review": false,
        "is_guess": false,
        "direction": "debit",
        "transaction_type": "expense"
      }
    }
  ],
  "skipped": []
}
```

Each transaction in the response is the **original object your backend
sent in, with all its original fields intact**, plus a `classification`
key. The backend never has to re-look-up a transaction by id in a separate
results array — everything it needs is right there on the record.

### `classification` fields

| Field               | Meaning                                                                 |
|---------------------|--------------------------------------------------------------------------|
| `predicted_class`   | One of the taxonomy classes in `taxonomy.py` (never null/omitted).       |
| `confidence`        | `0.0–1.0`. Below `0.6` → `needs_review = true`.                          |
| `alternative_class` | Second-best guess, or `null`.                                            |
| `reason_code`       | `MERCHANT_MATCH` \| `KEYWORD_MATCH` \| `RECURRING_PATTERN` \| `AMBIGUOUS` \| `INSUFFICIENT_DATA`. |
| `needs_review`      | `true` when confidence is below the review threshold.                    |
| `is_guess`          | `true` when there wasn't enough signal for a real match, so a best-effort heuristic guess was used instead (still always populated — see below). |
| `direction`         | `"debit"` (money out) or `"credit"` (money in), derived from amount sign.|
| `transaction_type`  | `"income"` \| `"expense"` \| `"transfer"` — the field your backend should filter on for cash-flow views. |

### Rate limiting

The `/api/classify` endpoint is rate-limited to **30 requests per minute per IP**
by default (configurable via `RATE_LIMIT_REQUESTS` and `RATE_LIMIT_WINDOW_SECONDS`).
Returns `429 Too Many Requests` with `Retry-After` header when exceeded.

### "Not enough information" never means "unclassified"

Every transaction always gets a `predicted_class` and a `transaction_type`.
When there's genuinely too little to go on, the service still returns its
best guess (using direction, recurrence, and counterparty signals) with a
low `confidence`, `is_guess: true`, and `needs_review: true` — so your
backend can render it, but flag it for a human or a follow-up rule instead
of dropping the transaction.

### Malformed transactions

A transaction missing an id (or not shaped like an object) can't be
normalized. Rather than silently dropping it, it's reported in the
top-level `skipped` array with a `reason`, and counted in
`summary.skipped_transactions`.

## Performance Optimizations (v0.2.0+)

| Area | Improvement |
|------|-------------|
| **Model** | Switched from 70B to 8B parameter models (llama-3.1-8b-instant on Groq, gemini-1.5-flash-8b, nemotron-3-nano-30b on NVIDIA NIM) — **~10x faster**, still accurate for classification |
| **Timeouts** | Hard 30s request timeout + 120s global middleware timeout prevents hung requests |
| **Retries** | Exponential backoff with jitter (1.5s, 3s, 6s) + respects provider `retry-after` headers |
| **Batching** | Adaptive batch sizes (5-25 based on workload), dynamic `max_tokens`, compact JSON |
| **Rate limiting** | Provider-aware TPM budgeting (Groq: 30k, Gemini: 15k RPM) with pacing |
| **Tokens** | Reduced prompt size (compact JSON, no indentation) ~40% token savings |
| **Workers** | Gunicorn with 2 Uvicorn workers for concurrent request handling |

## Project layout

| File              | Responsibility                                                        |
|-------------------|-------------------------------------------------------------------------|
| `taxonomy.py`     | The class list (`TxnClass`) + income/expense/transfer flow metadata.    |
| `normalizer.py`   | Maps heterogeneous Plaid/account schemas onto one `NormalizedTransaction`. |
| `classifier.py`   | `LLMClassifier` (Groq/Gemini/NVIDIA NIM/OpenRouter/Ollama) and `MockClassifier` (offline rules), sharing one `Prediction` contract and a shared no-transaction-left-behind fallback guesser. |
| `service.py`      | Orchestration: normalize → classify → merge back onto the original transaction + summary. This is what the backend should call/import. |
| `main.py`         | Thin FastAPI wrapper around `service.py` (the HTTP boundary) + a demo UI. Includes request timeout, rate limiting, and graceful shutdown. |
| `system_prompt.md`| The instructions sent to the LLM, including the output contract and an explicit instruction to ignore any embedded commands inside transaction text (prompt-injection hardening). |
| `dummy_data.py`   | Synthetic multi-schema demo data, including an adversarial prompt-injection test case. |
| `evaluator.py`    | Accuracy / precision / recall / confusion-matrix reporting against labeled data. |
| `demo.py`         | Runs the whole pipeline end-to-end against `dummy_data.py` and prints an accuracy report. |
| `llm_provider.py` | Provider configuration (Groq, Gemini, OpenRouter, Ollama) — swap models via env vars. |
| `logging_utils.py`| Structured JSON logging with per-request correlation IDs. |

## Security note

Transaction `description`/`merchant_name` text originates from external,
untrusted sources (merchants, banks). `system_prompt.md` explicitly
instructs the LLM never to follow instructions embedded in that text, and
`dummy_data.py` includes a regression test for this (`t031`). The mock
classifier is immune by construction (pure regex/keyword matching), but was
still audited to make sure no keyword rule could be tripped by attacker-
controlled text out of its intended context (e.g. "Salary" is only ever
inferred from *direction*, never from a bare keyword match).

## Extending the taxonomy

Edit `DEFAULT_TAXONOMY` in `taxonomy.py`. Give each class a `flow` of
`"income"`, `"expense"`, `"transfer"`, or `"ambiguous"` (resolved from the
transaction's actual direction at classification time) — this is what
drives `transaction_type` in the response.

## Configuration (Environment Variables)

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `groq` | `groq`, `gemini`, `nvidia`, `openrouter`, `ollama` |
| `LLM_API_KEY` | - | API key for the selected provider |
| `LLM_MODEL` | provider default | Override model name |
| `LLM_BASE_URL` | provider default | Override API base URL |
| `GEMINI_API_KEY` | - | Legacy: used if `LLM_API_KEY` not set and provider is `gemini` |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `REQUEST_TIMEOUT_SECONDS` | `120` | Global HTTP request timeout |
| `MAX_CHUNK_ATTEMPTS` | `3` | LLM chunk retry attempts |
| `RETRY_BACKOFF_SECONDS` | `1.5` | Base backoff for retries |
| `RATE_LIMIT_REQUESTS` | `30` | Requests per window per IP |
| `RATE_LIMIT_WINDOW_SECONDS` | `60` | Rate limit window in seconds |
| `PORT` | `8000` | HTTP port (set by platform) |
