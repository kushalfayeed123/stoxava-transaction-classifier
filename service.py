"""
Classification service (middleware layer).

This is the single entry point the main backend service should call. It
takes a raw Plaid response (or a bare list of transactions), and returns
each transaction *unmodified and complete*, merged with a `classification`
object the backend can filter/sort on (predicted_class, confidence,
direction, transaction_type/income-expense, needs_review, is_guess, etc.)
plus a batch-level summary.

Kept deliberately free of any FastAPI/HTTP concerns so it can be reused
from a queue worker, a CLI, or a different web framework without change.
"""

from __future__ import annotations

from typing import Any, Optional

from classifier import Classifier, Prediction
from normalizer import NormalizedTransaction, normalize_plaid_response
from taxonomy import TxnClass, DEFAULT_TAXONOMY


def _prediction_to_dict(p: Prediction) -> dict[str, Any]:
    return {
        "predicted_class": p.predicted_class,
        "confidence": round(p.confidence, 4),
        "alternative_class": p.alternative_class,
        "reason_code": p.reason_code,
        "needs_review": p.needs_review,
        "is_guess": p.is_guess,
        "direction": p.direction,          # "debit" | "credit"
        "transaction_type": p.transaction_type,  # "income" | "expense" | "transfer"
    }


def classify_plaid_response(
    payload: dict[str, Any] | list[Any],
    classifier: Classifier,
    *,
    taxonomy: list[TxnClass] = DEFAULT_TAXONOMY,
    sign_convention: str = "standard",
) -> dict[str, Any]:
    """Normalize -> classify -> re-attach to the original transaction.

    Returns a dict shaped as:
    {
      "backend": "MockClassifier" | "LLMClassifier",
      "summary": {...},
      "transactions": [
          {...every original field from the input transaction...,
           "classification": {...}},
          ...
      ]
    }

    Transactions that fail normalization (unparsable / missing id) are
    reported separately in `skipped` rather than silently dropped, so the
    backend can decide how to handle them (e.g. surface for manual review).
    """
    normalized, skipped = _normalize_with_skip_tracking(payload, sign_convention)

    predictions: list[Prediction] = classifier.classify_batch(normalized, taxonomy) if normalized else []
    predictions_by_id = {p.transaction_id: p for p in predictions}

    transactions_out: list[dict[str, Any]] = []
    income_total = 0.0
    expense_total = 0.0
    needs_review_count = 0
    guessed_count = 0

    for t in normalized:
        p = predictions_by_id.get(t.transaction_id)
        if p is None:
            # Defensive: classifier implementations must return one
            # Prediction per input transaction, but never let a missing
            # entry silently drop a transaction from the response.
            from classifier import _heuristic_guess, _enrich  # local import to avoid unused-at-module-load cost

            valid_names = {c.name for c in taxonomy}
            p = _enrich(
                t.transaction_id, *_heuristic_guess(t, valid_names),
                direction=t.direction, is_guess=True, taxonomy=taxonomy,
            )

        merged = dict(t.raw)  # complete original transaction, all fields preserved
        merged["classification"] = _prediction_to_dict(p)
        transactions_out.append(merged)

        if p.transaction_type == "income":
            income_total += abs(t.amount)
        elif p.transaction_type == "expense":
            expense_total += abs(t.amount)
        if p.needs_review:
            needs_review_count += 1
        if p.is_guess:
            guessed_count += 1

    return {
        "backend": type(classifier).__name__,
        "summary": {
            "total_transactions": len(transactions_out),
            "skipped_transactions": len(skipped),
            "income_total": round(income_total, 2),
            "expense_total": round(expense_total, 2),
            "needs_review_count": needs_review_count,
            "guessed_count": guessed_count,
        },
        "transactions": transactions_out,
        "skipped": skipped,
    }


def _normalize_with_skip_tracking(
    payload: dict[str, Any] | list[Any], sign_convention: str
) -> tuple[list[NormalizedTransaction], list[dict[str, Any]]]:
    """Wraps normalize_plaid_response so we can also report which raw
    records failed to normalize (id missing / totally malformed), instead
    of only logging them and losing track of the count/content."""
    from normalizer import normalize_transaction

    accounts_map: dict[str, dict[str, Any]] = {}
    transactions_list: list[Any] = []

    if isinstance(payload, dict):
        for acct in payload.get("accounts", []) or []:
            if isinstance(acct, dict) and acct.get("account_id") is not None:
                accounts_map[str(acct["account_id"])] = acct
        added = payload.get("added", [])
        if isinstance(added, list) and added:
            transactions_list = added
        elif isinstance(payload.get("transactions"), list):
            transactions_list = payload["transactions"]
    elif isinstance(payload, list):
        transactions_list = payload

    out: list[NormalizedTransaction] = []
    skipped: list[dict[str, Any]] = []
    for raw in transactions_list:
        if not isinstance(raw, dict):
            skipped.append({"reason": "not_an_object", "raw": raw})
            continue
        acct_id = raw.get("account_id")
        account_meta = accounts_map.get(str(acct_id), {}) if acct_id is not None else {}
        try:
            out.append(normalize_transaction(raw, account_meta=account_meta, sign_convention=sign_convention))
        except (ValueError, TypeError, KeyError) as exc:
            skipped.append({"reason": str(exc), "raw": raw})

    return out, skipped
