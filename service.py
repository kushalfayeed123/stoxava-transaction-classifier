"""
Classification service (middleware layer).

This is the single entry point the main backend service should call. It
takes a raw Plaid response (or a bare list of transactions), and returns
each transaction *unmodified and complete*, merged with a `classification`
object the backend can filter/sort on (predicted_class, confidence,
direction, transaction_type/income-expense, needs_review, is_guess, flow,
etc.) plus a batch-level summary.

Kept deliberately free of any FastAPI/HTTP concerns so it can be reused
from a queue worker, a CLI, or a different web framework without change.

Every call is tagged with a request id (see logging_utils) and logs its
full trail as structured JSON events: payload_parsed -> normalization_
complete -> [classifier's own classify_batch_start/.../complete events] ->
classification_request_complete. Grep/filter Render logs on one
request_id to see the whole path a given call took.

SCOPE NOTE on recurrence signals (amount/interval consistency, see
recurrence.py): they're only as good as the transaction list passed into
a single call here. On a full/historical Plaid sync that's plenty of
history to establish a counterparty's usual paycheck amount and cadence.
On routine incremental syncs, `payload` typically only carries a handful
of new "added"/"modified" records, so there often won't be enough same-call
history to judge a new deposit's pattern against -- REGULAR_PAYCHECK will fall
back to benefit-of-the-doubt (or the caller's own text/recurring-hint
signals) in that case, not because timing was verified. If the calling
backend can supply prior transactions for the same account/counterparty
alongside each incremental batch, pass them through classify_batch too
(they don't need separate predictions -- classify_batch already only
returns one Prediction per NormalizedTransaction, so filter down to just
the new ones after classifying, or thread the signal in another way).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from classifier import Classifier, Prediction, _enrich, _heuristic_guess
from logging_utils import ensure_request_id, log_event
from normalizer import (
    NormalizedTransaction,
    normalize_plaid_response,
    normalize_transaction,
)
from recurrence import compute_recurrence_signals
from taxonomy import TxnClass, DEFAULT_TAXONOMY

logger = logging.getLogger("transaction_classifier.service")


def _prediction_to_dict(p: Prediction) -> dict[str, Any]:
    return {
        "predicted_class": p.predicted_class,
        "confidence": round(p.confidence, 4),
        "alternative_class": p.alternative_class,
        "reason_code": p.reason_code,
        "needs_review": p.needs_review,
        "is_guess": p.is_guess,
        "direction": p.direction,  # "debit" | "credit"
        "transaction_type": p.transaction_type,  # "income" | "expense" | "transfer"
        "flow": p.flow,  # exactly which code path produced this classification
    }


def classify_plaid_response(
    payload: dict[str, Any] | list[Any],
    classifier: Classifier,
    *,
    taxonomy: list[TxnClass]=DEFAULT_TAXONOMY,
    sign_convention: str="standard",
) -> dict[str, Any]:
    """Normalize -> classify -> re-attach to the original transaction.

    Returns a dict shaped as:
    {
      "backend": "LLMClassifier",
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
    request_id = ensure_request_id()
    log_event(
        logger, "classification_request_started",
        request_id=request_id, backend=type(classifier).__name__, sign_convention=sign_convention,
    )

    normalized, skipped, diagnostics = _normalize_with_skip_tracking(payload, sign_convention)
    log_event(
        logger, "normalization_complete",
        normalized_count=len(normalized), skipped_count=len(skipped), **diagnostics,
    )

    predictions: list[Prediction] = classifier.classify_batch(normalized, taxonomy) if normalized else []
    predictions_by_id = {p.transaction_id: p for p in predictions}
    log_event(
        logger, "classifier_backend_returned",
        backend=type(classifier).__name__, prediction_count=len(predictions), input_count=len(normalized),
    )

    # Same signals classify_batch computes internally, recomputed here only
    # for the defensive fallback path below (a classifier implementation
    # returning no Prediction for some transaction). Cheap relative to an
    # LLM call, and keeps that path honoring the same REGULAR_PAYCHECK gate
    # instead of silently bypassing it.
    recurrence_signals = compute_recurrence_signals(normalized) if normalized else {}

    transactions_out: list[dict[str, Any]] = []
    income_total = 0.0
    expense_total = 0.0
    needs_review_count = 0
    guessed_count = 0
    missing_prediction_count = 0
    valid_names = {c.name for c in taxonomy}

    for t in normalized:
        p = predictions_by_id.get(t.transaction_id)
        if p is None:
            # Defensive: classifier implementations must return one
            # Prediction per input transaction, but never let a missing
            # entry silently drop a transaction from the response.
            missing_prediction_count += 1
            log_event(
                logger, "classifier_missing_prediction", level=logging.WARNING,
                transaction_id=t.transaction_id, backend=type(classifier).__name__,
            )
            cls, conf, reason = _heuristic_guess(t, valid_names, recurrence_signals.get(t.transaction_id))
            p = _enrich(
              t.transaction_id,
            cls,
            conf,
            None,  # explicitly pass None for alternative_class
            reason,  # now this correctly maps to reason_code
            direction=t.direction,
            is_guess=True,
            taxonomy=taxonomy,
            flow="service_missing_prediction_heuristic_fallback",
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

    summary = {
        "total_transactions": len(transactions_out),
        "skipped_transactions": len(skipped),
        "income_total": round(income_total, 2),
        "expense_total": round(expense_total, 2),
        "needs_review_count": needs_review_count,
        "guessed_count": guessed_count,
    }

    log_event(
        logger, "classification_request_complete",
        backend=type(classifier).__name__,
        missing_prediction_count=missing_prediction_count,
        **summary,
    )

    return {
        "backend": type(classifier).__name__,
        "summary": summary,
        "transactions": transactions_out,
        "skipped": skipped,
        # Only meaningful (and only worth reading) when total_transactions
        # is 0 -- explains exactly what was/wasn't found in the payload
        # instead of a flat "no transactions" message.
        "diagnostics": diagnostics,
    }


def _normalize_with_skip_tracking(
    payload: dict[str, Any] | list[Any], sign_convention: str
) -> tuple[list[NormalizedTransaction], list[dict[str, Any]], dict[str, Any]]:
    """Wraps normalize_plaid_response so we can also report which raw
    records failed to normalize (id missing / totally malformed), instead
    of only logging them and losing track of the count/content.

    Also returns a `diagnostics` dict describing exactly what was found in
    the payload, so a "0 transactions" result can explain *why* instead of
    failing silently."""
    accounts_map: dict[str, dict[str, Any]] = {}
    transactions_list: list[Any] = []
    source_field: Optional[str] = None
    diagnostics: dict[str, Any] = {}

    # Defensive auto-unwrap: it's an easy, common mistake to paste/send an
    # already-{"transactions": {...}}-wrapped body as the *value* of
    # `transactions` (double wrapping). If the given payload has no
    # "accounts"/"added" of its own but its "transactions" key holds a dict
    # that looks like the real Plaid envelope, unwrap it one level rather
    # than reporting zero transactions.
    if (
        isinstance(payload, dict)
        and "accounts" not in payload
        and "added" not in payload
        and isinstance(payload.get("transactions"), dict)
    ):
        inner = payload["transactions"]
        if "accounts" in inner or "added" in inner or isinstance(inner.get("transactions"), list):
            diagnostics["auto_unwrapped"] = True
            log_event(logger, "payload_auto_unwrapped", level=logging.INFO)
            payload = inner

    if isinstance(payload, dict):
        diagnostics["top_level_keys"] = list(payload.keys())
        for acct in payload.get("accounts", []) or []:
            if isinstance(acct, dict) and acct.get("account_id") is not None:
                accounts_map[str(acct["account_id"])] = acct

        added = payload.get("added", [])
        modified = payload.get("modified", [])
        diagnostics["added_count"] = len(added) if isinstance(added, list) else None
        diagnostics["modified_count"] = len(modified) if isinstance(modified, list) else None

        # A Plaid /transactions/sync page can carry new transactions in
        # "added" AND status changes (e.g. pending -> posted) in
        # "modified" -- both represent real transactions worth
        # classifying, so merge them rather than only reading "added".
        merged_list: list[Any] = []
        if isinstance(added, list):
            merged_list.extend(added)
        if isinstance(modified, list):
            merged_list.extend(modified)
        if merged_list:
            transactions_list = merged_list
            source_field = "added+modified"
        elif isinstance(payload.get("transactions"), list) and payload["transactions"]:
            transactions_list = payload["transactions"]
            source_field = "transactions"
    elif isinstance(payload, list):
        transactions_list = payload
        source_field = "bare_list"
        diagnostics["top_level_keys"] = None

    diagnostics["source_field_used"] = source_field
    diagnostics["candidate_count"] = len(transactions_list)

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
            log_event(
                logger, "transaction_normalization_failed", level=logging.WARNING,
                reason=str(exc),
            )

    return out, skipped, diagnostics
