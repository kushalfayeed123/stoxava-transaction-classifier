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

- Set `GEMINI_API_KEY` to use the real LLM backend (Gemini). Without it (or
  if the LLM client fails to initialize) the service automatically falls
  back to `MockClassifier`, a deterministic rule engine, so the service
  never fails to start and always returns a result.
- `GET /health` → `{"status": "ok", "backend": "MockClassifier"}`
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
an optional `GEMINI_API_KEY`, so any free container host works. Two ready
to go:

### Render (recommended — simplest, has a free web-service tier)

1. Push this folder to a GitHub repo.
2. On [render.com](https://render.com): **New → Blueprint**, point it at
   the repo. Render reads `render.yaml` in this project automatically and
   creates the service — no manual config needed.
3. (Optional) In the service's **Environment** tab, add `GEMINI_API_KEY` if
   you want the real LLM backend; otherwise it runs `MockClassifier`.
4. You'll get a URL like `https://transaction-classifier-xxxx.onrender.com`
   — share `.../docs` with other devs.

Free-tier caveat: the instance sleeps after ~15 minutes idle and takes a
few seconds to cold-start on the next request — fine for testing, just hit
`/health` once before a live demo to warm it up.

### Fly.io (alternative — no sleep on the free allowance)

```bash
brew install flyctl   # or see fly.io/docs/hands-on/install-flyctl
fly launch --no-deploy   # detects the Dockerfile, uses fly.toml in this repo
fly secrets set GEMINI_API_KEY=...   # optional
fly deploy
```

Either way, once it's live and validated, the same `Dockerfile` is what
you'd hand to the VPS (`docker build -t transaction-classifier . && docker
run -p 8000:8000 --env-file .env transaction-classifier`) — no rewrite
needed for the move.

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

## Project layout

| File              | Responsibility                                                        |
|-------------------|-------------------------------------------------------------------------|
| `taxonomy.py`     | The class list (`TxnClass`) + income/expense/transfer flow metadata.    |
| `normalizer.py`   | Maps heterogeneous Plaid/account schemas onto one `NormalizedTransaction`. |
| `classifier.py`   | `LLMClassifier` (Gemini) and `MockClassifier` (offline rules), sharing one `Prediction` contract and a shared no-transaction-left-behind fallback guesser. |
| `service.py`      | Orchestration: normalize → classify → merge back onto the original transaction + summary. This is what the backend should call/import. |
| `main.py`         | Thin FastAPI wrapper around `service.py` (the HTTP boundary) + a demo UI. |
| `system_prompt.md`| The instructions sent to Gemini, including the output contract and an explicit instruction to ignore any embedded commands inside transaction text (prompt-injection hardening). |
| `dummy_data.py`   | Synthetic multi-schema demo data, including an adversarial prompt-injection test case. |
| `evaluator.py`    | Accuracy / precision / recall / confusion-matrix reporting against labeled data. |
| `demo.py`         | Runs the whole pipeline end-to-end against `dummy_data.py` and prints an accuracy report. |

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
