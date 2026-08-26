"""
Classification backends.

LLMClassifier calls a Gemini model using the system prompt in
system_prompt.md via the google-genai SDK -- requires GEMINI_API_KEY.
MockClassifier classifies deterministically (Plaid category data, then
heuristics) with no external API calls -- used when GEMINI_API_KEY isn't
configured, so local dev/demos/tests never require a live key.

Every call returns the `Prediction` dataclass so the UI and evaluator never
need to know how a result was produced. Every Prediction also carries a
`flow` string identifying exactly which code path produced it (e.g. "llm",
"llm_override_REGULAR_PAYCHECK", "mock_heuristic", "chunk_error_plaid_fallback")
-- both logged at classification time and returned in the API response, so
"why did this transaction get this class" is always answerable without
guessing.

Design principle: **every transaction always gets a predicted_class.**
Nothing is ever left unclassified. When there truly isn't enough signal to
make a confident call, the classifier falls back to `_heuristic_guess`,
which makes the best possible guess from direction / recurrence /
counterparty signals and reports a low (but non-zero, non-arbitrary)
confidence score plus `is_guess=True` so downstream consumers can tell
"classified" apart from "guessed."

Second design principle, for amount/timing consistency: REGULAR_PAYCHECK is a
claim about a *pattern* (same amount, same cadence), not just "looks like a
paycheck." A deposit from a known payroll counterparty that's the wrong
size or off-schedule compared to that counterparty's own history is NOT a
regular paycheck, even if the merchant name or a recurring-hint flag says
otherwise. `recurrence.py` computes that pattern-match signal per
transaction, and it acts as a hard gate here: once a counterparty has
enough history to judge, REGULAR_PAYCHECK is only ever assigned (heuristically,
by Plaid's own category, or as a correction to an LLM prediction) when the
signal agrees.
"""

from __future__ import annotations
import time

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from openai import APIStatusError, RateLimitError

from llm_provider import ProviderConfig
from logging_utils import ensure_request_id, log_event
from normalizer import NormalizedTransaction
from recurrence import RecurrenceSignal, compute_recurrence_signals
from taxonomy import TxnClass, taxonomy_to_prompt_block, resolve_flow_type, \
    Direction
    
import collections

SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent / "system_prompt.md"

logger = logging.getLogger("transaction_classifier.classifier")

# --------------------------------------------------------------------------
# Plaid's own `personal_finance_category` as a second-tier signal.
#
# Real Plaid data almost always carries a `personal_finance_category` with
# `primary`/`detailed` values, and it's frequently right even when a merchant
# name isn't in any keyword list. This is checked *before* the last-resort
# heuristic guess, since it's real categorization data, not a blind guess.
# --------------------------------------------------------------------------

_PFC_DETAILED_MAP: dict[str, str] = {
    # -- expense side --
    "FOOD_AND_DRINK_GROCERIES": "Groceries",
    "FOOD_AND_DRINK_RESTAURANT": "Dining",
    "FOOD_AND_DRINK_COFFEE": "Dining",
    "FOOD_AND_DRINK_FAST_FOOD": "Dining",
    "FOOD_AND_DRINK_VENDING_MACHINES": "Dining",
    "FOOD_AND_DRINK_BEER_WINE_AND_LIQUOR": "Groceries",
    "GENERAL_MERCHANDISE_SUPERSTORES": "Groceries",
    "GENERAL_MERCHANDISE_ONLINE_MARKETPLACES": "Shopping",
    "GENERAL_MERCHANDISE_DEPARTMENT_STORES": "Shopping",
    "GENERAL_MERCHANDISE_ELECTRONICS": "Shopping",
    "GENERAL_MERCHANDISE_CLOTHING_AND_ACCESSORIES": "Shopping",
    "RENT_AND_UTILITIES_RENT": "Housing",
    "RENT_AND_UTILITIES_GAS_AND_ELECTRICITY": "Utilities",
    "RENT_AND_UTILITIES_INTERNET_AND_CABLE": "Utilities",
    "RENT_AND_UTILITIES_TELEPHONE": "Utilities",
    "RENT_AND_UTILITIES_WATER": "Utilities",
    "RENT_AND_UTILITIES_SEWAGE_AND_WASTE_MANAGEMENT": "Utilities",
    "TRANSPORTATION_GAS": "Transportation",
    "TRANSPORTATION_PARKING": "Transportation",
    "TRANSPORTATION_PUBLIC_TRANSIT": "Transportation",
    "TRANSPORTATION_TAXIS_AND_RIDE_SHARES": "Transportation",
    "TRAVEL_FLIGHTS": "Travel",
    "TRAVEL_LODGING": "Travel",
    "TRAVEL_RENTAL_CARS": "Travel",
    "MEDICAL_PRIMARY_CARE": "Healthcare",
    "MEDICAL_DENTAL_CARE": "Healthcare",
    "MEDICAL_PHARMACIES_AND_SUPPLEMENTS": "Healthcare",
    "MEDICAL_VETERINARY_SERVICES": "Healthcare",
    "ENTERTAINMENT_MOVIES_AND_DVDS": "Entertainment",
    "ENTERTAINMENT_MUSIC_AND_AUDIO": "Entertainment",
    "ENTERTAINMENT_TV_AND_MOVIES": "Subscriptions",
    "ENTERTAINMENT_SPORTING_EVENTS_AMUSEMENT_PARKS_AND_MUSEUMS": "Entertainment",
    "GENERAL_SERVICES_INSURANCE": "Insurance",
    "LOAN_PAYMENTS_STUDENT_LOAN_PAYMENT": "Debt Payments",
    "LOAN_PAYMENTS_PERSONAL_LOAN_PAYMENT": "Debt Payments",
    "LOAN_PAYMENTS_CAR_PAYMENT": "Debt Payments",
    "LOAN_PAYMENTS_MORTGAGE_PAYMENT": "Housing",
    "BANK_FEES_OVERDRAFT_FEES": "Other",
    "BANK_FEES_ATM_FEES": "Other",
    "BANK_FEES_INTEREST_CHARGE": "Other",
    "TRANSFER_OUT_WITHDRAWAL": "Transfers / Excluded",
    "TRANSFER_IN_CASH_ADVANCES_AND_LOANS": "LOAN",

    # -- income side (names must match _DEFAULT_INCOME_TAXONOMY) --
    "INCOME_WAGES": "REGULAR_PAYCHECK",
    "INCOME_RETIREMENT_PENSION": "OTHER_INCOME",
    "INCOME_DIVIDENDS": "OTHER_INCOME",
    "INCOME_TAX_REFUND": "REFUND",
    "TRANSFER_IN_DEPOSIT": "TRANSFER",
}

_PFC_PRIMARY_MAP: dict[str, str] = {
    "FOOD_AND_DRINK": "Dining",
    "GENERAL_MERCHANDISE": "Shopping",
    "RENT_AND_UTILITIES": "Housing",
    "TRANSPORTATION": "Transportation",
    "TRAVEL": "Travel",
    "MEDICAL": "Healthcare",
    "ENTERTAINMENT": "Entertainment",
    "LOAN_PAYMENTS": "Debt Payments",
    "BANK_FEES": "Other",
    "INCOME": "REGULAR_PAYCHECK",
    "TRANSFER_IN": "TRANSFER",
    "TRANSFER_OUT": "Transfers / Excluded",
}


def _match_plaid_category(
    t: NormalizedTransaction,
    valid_names: set[str],
    signal: Optional[RecurrenceSignal]=None,
) -> Optional[tuple[str, float, str]]:
    """Returns (predicted_class, confidence, reason_code) from Plaid's own
    category fields, or None if nothing usable is present.

    When Plaid's own category points at REGULAR_PAYCHECK but the amount/interval
    signal says this deposit doesn't match this counterparty's established
    pattern, we don't trust Plaid's category for that specific claim --
    fall through to OTHER_W2 instead, which still says "this is W2 income"
    without asserting it's the *regular, predictable* paycheck.
    """
    detailed = (t.plaid_category_detailed or "").upper()
    if detailed in _PFC_DETAILED_MAP:
        cls = _PFC_DETAILED_MAP[detailed]
        cls = _guard_REGULAR_PAYCHECK(cls, signal, valid_names)
        if cls in valid_names:
            return cls, 0.78, "PLAID_CATEGORY_MATCH"

    for token in t.plaid_category:
        primary = token.upper().replace(" ", "_")
        if primary in _PFC_PRIMARY_MAP:
            cls = _PFC_PRIMARY_MAP[primary]
            cls = _guard_REGULAR_PAYCHECK(cls, signal, valid_names)
            if cls in valid_names:
                return cls, 0.65, "PLAID_CATEGORY_MATCH"

    # Legacy free-text category lists -- light keyword scan over raw tokens.
    joined = " ".join(t.plaid_category).lower()
    if "subscription" in joined and "Subscriptions" in valid_names:
        return "Subscriptions", 0.7, "PLAID_CATEGORY_MATCH"
    if any(k in joined for k in ("grocery", "groceries", "food and beverage store")) and "Groceries" in valid_names:
        return "Groceries", 0.7, "PLAID_CATEGORY_MATCH"
    if "restaurant" in joined and "Dining" in valid_names:
        return "Dining", 0.7, "PLAID_CATEGORY_MATCH"

    return None


def _guard_REGULAR_PAYCHECK(
    candidate_class: str,
    signal: Optional[RecurrenceSignal],
    valid_names: set[str],
) -> str:
    """Hard gate: only ever let REGULAR_PAYCHECK through when there isn't yet
    enough history to judge (group_size < 2, benefit of the doubt on a
    first-seen deposit) or when the amount+interval signal agrees this
    deposit matches the counterparty's established pattern. Otherwise
    redirect to OTHER_W2 (still W2 income, just not asserted "regular")."""
    if candidate_class != "REGULAR_PAYCHECK":
        return candidate_class
    if signal is None or not signal.has_enough_history or signal.is_regular_candidate:
        return candidate_class
    return "OTHER_W2" if "OTHER_W2" in valid_names else candidate_class


@dataclass
class Prediction:
    transaction_id: str
    predicted_class: str
    confidence: float
    alternative_class: Optional[str]
    reason_code: Optional[str]

    # -- enrichment fields required by downstream backend consumers --
    direction: str = "debit"  # "debit" | "credit" (derived from amount sign)
    transaction_type: str = "expense"  # "income" | "expense" | "transfer"
    is_guess: bool = False  # True when confidence comes from the
                                       # fallback heuristic rather than a
                                       # real merchant/category/LLM match
    # Exactly which code path produced this prediction -- see module
    # docstring for the full set of flow names. Always set; "unknown" only
    # if a caller constructs a Prediction directly without going through
    # _enrich (shouldn't happen in normal operation).
    flow: str = "unknown"

    NEEDS_REVIEW_THRESHOLD = 0.6

    @property
    def needs_review(self) -> bool:
        return self.confidence < self.NEEDS_REVIEW_THRESHOLD


class Classifier:
    """Interface implemented by the LLM and Mock backends."""
    
    def classify_batch(
        self,
        transactions: list[NormalizedTransaction],
        taxonomy: list[TxnClass],
    ) -> list[Prediction]:
        raise NotImplementedError

# --------------------------------------------------------------------------
# Shared fallback: never leave a transaction with no guess at all.
# --------------------------------------------------------------------------


def _heuristic_guess(
    t: NormalizedTransaction,
    valid_names: set[str],
    signal: Optional[RecurrenceSignal]=None,
) -> tuple[str, float, str]:
    text = f"{t.merchant_name or ''} {t.description}".lower()

    if t.direction == "credit":
        # Strong textual signal -- but only trust it for REGULAR_PAYCHECK when
        # there's no established pattern to contradict it yet, or the
        # pattern agrees. A same-source deposit that's the wrong amount or
        # off-schedule is not a "regular" paycheck no matter what the
        # merchant text says.
        looks_like_payroll = any(k in text for k in ("payroll", "salary", "direct dep", "paycheck"))
        pattern_says_regular = signal is None or not signal.has_enough_history or signal.is_regular_candidate

        if looks_like_payroll and "REGULAR_PAYCHECK" in valid_names and pattern_says_regular:
            return "REGULAR_PAYCHECK", 0.6, "MERCHANT_MATCH"
        if looks_like_payroll and "OTHER_W2" in valid_names and not pattern_says_regular:
            # Same-employer income signal is real, but the pattern broke --
            # report it as W2 income without claiming regularity.
            return "OTHER_W2", 0.45, "AMOUNT_INTERVAL_MISMATCH"

        if t.is_recurring_hint and "REGULAR_PAYCHECK" in valid_names and pattern_says_regular:
            return "REGULAR_PAYCHECK", 0.55, "RECURRING_PATTERN"
        if t.is_recurring_hint and "OTHER_W2" in valid_names and not pattern_says_regular:
            return "OTHER_W2", 0.4, "AMOUNT_INTERVAL_MISMATCH"

        if "UNCLASSIFIED" in valid_names:
            return "UNCLASSIFIED", 0.2, "INSUFFICIENT_DATA"

    if t.is_recurring_hint and "Subscriptions" in valid_names:
        return "Subscriptions", 0.55, "RECURRING_PATTERN"
    if t.counterparty_type == "person" and "Needs Review" in valid_names:
        return "Needs Review", 0.5, "AMBIGUOUS"
    if "Needs Review" in valid_names and t.direction == "debit":
        return "Needs Review", min(0.45, 0.15 + (0.08 if t.merchant_name else 0)
                                    +(0.08 if t.plaid_category else 0)), "INSUFFICIENT_DATA"

    fallback = "Uncategorized" if "Uncategorized" in valid_names else next(iter(valid_names), "Uncategorized")
    return fallback, 0.25, "INSUFFICIENT_DATA"


def _enrich(
    transaction_id: str,
    predicted_class: str,
    confidence: float,
    alternative_class: Optional[str],
    reason_code: Optional[str]="DEFAULT_CLASSIFICATION",
    *,
    direction: Direction,
    is_guess: bool,
    taxonomy: list[TxnClass],
    flow: str="unknown",
) -> Prediction:
    transaction_type = resolve_flow_type(predicted_class, direction, taxonomy)
    return Prediction(
        transaction_id=transaction_id,
        predicted_class=predicted_class,
        confidence=confidence,
        alternative_class=alternative_class,
        reason_code=reason_code,
        direction=direction,
        transaction_type=transaction_type,
        is_guess=is_guess,
        flow=flow,
    )


def _classify_via_plaid_or_heuristic(
    t: NormalizedTransaction,
    valid_names: set[str],
    signal: Optional[RecurrenceSignal],
    taxonomy: list[TxnClass],
    flow_if_plaid: str,
    flow_if_heuristic: str,
    log_level: int=logging.DEBUG,
) -> Prediction:
    """Shared "try Plaid category, else heuristic guess" path used by every
    fallback site (chunk error, LLM missing a transaction, LLM returning an
    invalid class, MockClassifier). Centralized so the flow name assigned
    always matches which branch actually fired, and so every fallback gets
    logged the same way."""
    plaid_match = _match_plaid_category(t, valid_names, signal)
    if plaid_match is not None:
        cls, conf, reason = plaid_match
        flow = flow_if_plaid
        is_guess = False   # real categorization data, not a guess
    else:
        cls, conf, reason = _heuristic_guess(t, valid_names, signal)
        flow = flow_if_heuristic
        is_guess = True

    log_event(
        logger, "transaction_classified", level=log_level,
        transaction_id=t.transaction_id, flow=flow,
        predicted_class=cls, confidence=conf, reason_code=reason,
    )
    return _enrich(
        t.transaction_id, cls, conf, None, reason,
        direction=t.direction, is_guess=is_guess, taxonomy=taxonomy, flow=flow,
    )

# --------------------------------------------------------------------------
# Mock classifier -- no external API calls, fully deterministic. Used when
# GEMINI_API_KEY isn't configured (local dev, demos, tests) so the service
# never fails to start or refuses to classify.
# --------------------------------------------------------------------------


class MockClassifier(Classifier):

    def classify_batch(
        self,
        transactions: list[NormalizedTransaction],
        taxonomy: list[TxnClass],
    ) -> list[Prediction]:
        ensure_request_id()
        signals = compute_recurrence_signals(transactions)
        valid_names = {c.name for c in taxonomy}

        log_event(logger, "classify_batch_start", backend="MockClassifier", total_transactions=len(transactions))

        out = [
            _classify_via_plaid_or_heuristic(
                t, valid_names, signals.get(t.transaction_id), taxonomy,
                flow_if_plaid="mock_plaid_category",
                flow_if_heuristic="mock_heuristic",
            )
            for t in transactions
        ]

        log_event(
            logger, "classify_batch_complete", backend="MockClassifier",
            total_transactions=len(transactions),
            needs_review_count=sum(1 for p in out if p.needs_review),
        )
        return out

# --------------------------------------------------------------------------
# LLM-backed classifier (production path using Gemini Free Tier)
# --------------------------------------------------------------------------
# Retry settings for transient failures (SSL resets, rate limits, timeouts).


MAX_CHUNK_ATTEMPTS = 3  # total tries per chunk, incl. the first
RETRY_BACKOFF_SECONDS = 2.0  # exponential: 2s, 4s

_RETRY_HINT_RE = re.compile(r"[Pp]lease try again in ([\d.]+)s")


def _retry_delay(attempt: int, exc: Exception) -> float:
    """Delay honoring the server's own hint when available."""
    # Source 1: SDK-parsed header attribute (present on some versions)
    ra = getattr(exc, "retry_after", None)
    if ra:
        return float(ra) + 1.0

    # Source 2: raw response headers
    headers = getattr(getattr(exc, "response", None), "headers", None) or {}
    ra = headers.get("retry-after")
    if ra:
        try:
            return float(ra) + 1.0
        except ValueError:
            pass

    # Source 3: parse Groq's human-readable hint from the error message
    m = _RETRY_HINT_RE.search(str(exc))
    if m:
        return float(m.group(1)) + 1.0

    return RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))

_VALID_LLM_REASON_CODES: frozenset[str] = frozenset({
            "LLM_MERCHANT_KNOWLEDGE",
            "LLM_CONTEXTUAL_INFERENCE",
            "LLM_RECURRING_PATTERN",
            "LLM_AMBIGUOUS",
            "LLM_INSUFFICIENT_DATA",
        })


def _normalize_reason_code(raw: Any) -> str:
    """Map any LLM-returned reason code onto the valid LLM_* vocabulary."""
    code = str(raw or "").strip()
    if code in _VALID_LLM_REASON_CODES:
        return code
    if code:
        log_event(logger, "llm_reason_code_normalized", level=logging.DEBUG,
                raw_code=code[:60])
    return "LLM_AMBIGUOUS"

def _parse_llm_confidence(p: dict) -> Optional[float]:
        """Safely extract and clamp confidence. Returns None if missing or
        unconvertible -- callers fall back per-transaction rather than raising
        (which would burn chunk retries)."""
        raw = p.get("confidence")
        try:
            conf = float(raw)
        except (TypeError, ValueError):
            return None
        return min(max(conf, 0.0), 1.0)

class LLMClassifier(Classifier):

    def __init__(self, batch_size: int=5,
                    provider_config: Optional["ProviderConfig"]=None):
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError(
                    "The 'openai' package is required for LLMClassifier. "
                    "Install with: uv add openai"
                ) from exc
            from llm_provider import ProviderConfig  # local import avoids cycle at module load

            cfg = provider_config or ProviderConfig()

            if not SYSTEM_PROMPT_PATH.exists():
                raise FileNotFoundError(
                    f"system_prompt.md not found at {SYSTEM_PROMPT_PATH}. "
                    "LLMClassifier requires this file alongside classifier.py."
                )

            extra_kwargs: dict = {}
           
            if cfg.provider == "openrouter":
                extra_kwargs["default_headers"] = {
                    "HTTP-Referer": "https://stoxava.local",
                    "X-Title": "Stoxava Transaction Classifier",
                }

            self._client = OpenAI(
                api_key=cfg.api_key,
                base_url=cfg.base_url,
                timeout=60.0,
                max_retries=0,  # we own retries at the chunk level; avoid double-retrying
                ** extra_kwargs,
            )
            self._provider = cfg.provider
            self._model = cfg.model
            self._batch_size = batch_size
            self._system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
            self._recent_sends: collections.deque[float] = collections.deque()
        
  


    def _pace(self):
        """Sleep until sending another ~4k-token request stays under TPM."""
        TPM_BUDGET = 8000
        TOKENS_PER_REQUEST = 4200  # slightly overestimate for safety
        now = time.monotonic()
        while self._recent_sends and now - self._recent_sends[0] > 60:
            self._recent_sends.popleft()
        if self._recent_sends:
            # tokens already spent in this window vs. what remains
            used = len(self._recent_sends) * TOKENS_PER_REQUEST
            if used + TOKENS_PER_REQUEST > TPM_BUDGET:
                sleep_for = 60 - (now - self._recent_sends[0]) + 1.0
                log_event(logger, "llm_pacing", level=logging.DEBUG,
                            sleeping_seconds=round(sleep_for, 1))
                time.sleep(max(sleep_for, 0))
        self._recent_sends.append(time.monotonic())

    def classify_batch(self, transactions, taxonomy):
        ensure_request_id()
        signals = compute_recurrence_signals(transactions)

        num_chunks = ((len(transactions) + self._batch_size - 1)
                      // self._batch_size) if transactions else 0
        log_event(logger, "classify_batch_start", backend="LLMClassifier",
                  provider=self._provider, model=self._model,
                  total_transactions=len(transactions),
                  batch_size=self._batch_size, chunk_count=num_chunks)

        results: list[Prediction] = []
        for idx, start in enumerate(range(0, len(transactions), self._batch_size)):
            chunk = transactions[start: start + self._batch_size]
            log_event(logger, "llm_chunk_start", chunk_index=idx,
                      chunk_size=len(chunk), of_chunks=num_chunks)

            self._pace()
            attempt = 0
            while True:
                try:
                    chunk_results = self._classify_chunk(chunk, taxonomy, signals)
                    log_event(logger, "llm_chunk_success", chunk_index=idx,
                              chunk_size=len(chunk))
                    results.extend(chunk_results)
                    break
                except RateLimitError as exc:
                    attempt += 1
                    if attempt >= MAX_CHUNK_ATTEMPTS:
                        log_event(logger, "llm_chunk_failed_falling_back",
                                  level=logging.WARNING, chunk_index=idx,
                                  attempts=attempt, error_type="RateLimitError",
                                  error=str(exc)[:300])
                        results.extend(self._fallback_chunk(chunk, taxonomy, signals))
                        break
                    sleep_s = _retry_delay(attempt, exc)  # e.g. 24s — actually waits
                    log_event(logger, "llm_chunk_retrying", level=logging.WARNING,
                              chunk_index=idx, attempt=attempt,
                              max_attempts=MAX_CHUNK_ATTEMPTS,
                              retry_in_seconds=sleep_s)
                    time.sleep(sleep_s)
                except APIStatusError as exc:
                    # permanent config/request errors -- never retry
                    log_event(logger, "llm_chunk_failed_falling_back",
                              level=logging.WARNING, chunk_index=idx,
                              permanent=True, status_code=exc.status_code,
                              error=str(exc)[:500])
                    results.extend(self._fallback_chunk(chunk, taxonomy, signals))
                    break
                except Exception as exc:
                    attempt += 1
                    if attempt >= MAX_CHUNK_ATTEMPTS:
                        log_event(logger, "llm_chunk_failed_falling_back",
                                  level=logging.WARNING, chunk_index=idx,
                                  attempts=attempt, error_type=type(exc).__name__,
                                  error=str(exc)[:500], exc_info=True)
                        results.extend(self._fallback_chunk(chunk, taxonomy, signals))
                        break
                    sleep_s = _retry_delay(attempt, exc)
                    time.sleep(sleep_s)

            # Pace between chunks: with ~4k-token requests on an 8k TPM budget,
            # a short gap keeps consecutive chunks from colliding in-window.
            time.sleep(2.0)

        log_event(logger, "classify_batch_complete", backend="LLMClassifier",
                  provider=self._provider,
                  total_transactions=len(transactions),
                  llm_count=sum(1 for p in results if p.flow == "llm"),
                  override_count=sum(1 for p in results if p.flow == "llm_override_REGULAR_PAYCHECK"),
                  fallback_count=sum(1 for p in results if p.is_guess),
                  needs_review_count=sum(1 for p in results if p.needs_review))
        return results

    @staticmethod
    def _fallback_chunk(
        chunk: list[NormalizedTransaction],
        taxonomy: list[TxnClass],
        signals: dict[str, RecurrenceSignal],
    ) -> list[Prediction]:
        valid_names = {c.name for c in taxonomy}
        return [
            _classify_via_plaid_or_heuristic(
                t, valid_names, signals.get(t.transaction_id), taxonomy,
                flow_if_plaid="chunk_error_plaid_fallback",
                flow_if_heuristic="chunk_error_heuristic_fallback",
                log_level=logging.INFO,
            )
            for t in chunk
        ]


    @staticmethod
    def _build_examples() -> list[dict]:
        """Few-shot examples shown to the model. Placeholder IDs ONLY --
        never use a real transaction_id from the current chunk here, or the
        model may echo it and collide/drop its real predictions."""
        return [
            {
                "transaction_id": "<first_transaction_id>",
                "predicted_class": "<class name from taxonomy>",
                "confidence": 0.93,
                "alternative_class": None,
                "reason_code": "LLM_MERCHANT_KNOWLEDGE",
            },
            {
                "transaction_id": "<second_transaction_id>",
                "predicted_class": "<best guess from taxonomy>",
                "confidence": 0.55,
                "alternative_class": "<second-best guess from taxonomy>",
                "reason_code": "LLM_INSUFFICIENT_DATA",
            },
        ]

    def _classify_chunk(self, chunk, taxonomy, signals):
      

        annotated = []
        for t in chunk:
            d = t.to_dict()
            sig = signals.get(t.transaction_id)

            compact = {
                "transaction_id": t.transaction_id,
                "description": d.get("description") or d.get("name") or d.get("merchant_name"),
                "amount": d.get("amount"),
                "date": d.get("date"),
                "direction": t.direction,
                # include whatever category field your normalizer actually sets,
                # if present in the dict:
                **({k: v for k, v in d.items()
                    if k in ("category", "plaid_category", "merchant_name")
                    and v is not None}),
            }
            if sig is not None:
                compact["recurrence_signal"] = sig.to_prompt_fragment()
            annotated.append(compact)

        user_content = (
            taxonomy_to_prompt_block(taxonomy)
            + "\nRespond with ONLY a compact JSON object (no indentation, "
              "one prediction per line) of this exact shape:\n"
            + json.dumps({"predictions": self._build_examples()})
            + "\nTRANSACTIONS:\n"
            + json.dumps(annotated, indent=2)
        )

        kwargs: dict = {}
        if self._provider != "ollama":
            kwargs["response_format"] = {"type": "json_object"}

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
            max_tokens=2000,
            **kwargs,
        )
        log_event(logger, "llm_prompt_size", level=logging.DEBUG,
                  taxonomy_block_chars=len(taxonomy_to_prompt_block(taxonomy)),
                  system_prompt_chars=len(self._system_prompt),
                  transactions_json_chars=len(json.dumps(annotated)),
                  total_chars=len(user_content))
        return self._parse_response(response, chunk, taxonomy, signals)

   

    
    @staticmethod
    def _parse_response(
        response: Any,
        chunk: list[NormalizedTransaction],
        taxonomy: list[TxnClass],
        signals: dict[str, RecurrenceSignal],
    ) -> list[Prediction]:
        """Parse an LLM chat completion into Predictions for every transaction
        in `chunk`.

        Takes the full API response (not just text) so we can inspect
        finish_reason for truncation. Never raises: any unparsable or
        incomplete output degrades per-transaction to the Plaid-category /
        heuristic fallback path.

        NOTE: callers should now pass the raw response object:
            return self._parse_response(response, chunk, taxonomy, signals)
        """
        valid_names = {c.name for c in taxonomy}

        # ---- Extract text and detect truncation -------------------------
        choice = response.choices[0]
        text = choice.message.content or ""
        finish_reason = getattr(choice, "finish_reason", None)

        if finish_reason == "length":
            log_event(
                logger, "llm_response_truncated", level=logging.WARNING,
                chunk_size=len(chunk),
                hint="reduce batch_size or raise max_tokens",
            )
            # Don't bail -- the per-id loop below falls back for any
            # transactions missing from the truncated output.

        # ---- Layered JSON extraction ------------------------------------
        # Layer 1: strip markdown fences (existing behavior).
        cleaned = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()

        raw_predictions = None
        try:
            raw_predictions = json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        if raw_predictions is None:
            # Layer 2 (reasoning models like gpt-oss): pull the outermost JSON
            # structure out of surrounding commentary. Try arrays before
            # objects -- a truncated response ending mid-object is hopeless,
            # but one ending mid-array may still yield a valid partial result.
            for opener, closer in (("[", "]"), ("{", "}")):
                start = cleaned.find(opener)
                end = cleaned.rfind(closer)
                if start != -1 and end > start:
                    try:
                        raw_predictions = json.loads(cleaned[start:end + 1])
                        break
                    except json.JSONDecodeError:
                        continue

        if raw_predictions is None:
            log_event(
                logger, "llm_response_unparsable", level=logging.WARNING,
                chunk_size=len(chunk), finish_reason=finish_reason,
                response_preview=text[:200],
            )
            return LLMClassifier._fallback_chunk(chunk, taxonomy, signals)

        # ---- Normalize shape --------------------------------------------
        # The model may wrap the array in a dict under various keys.
        if isinstance(raw_predictions, dict):
            for key in ("predictions", "transactions", "results", "data"):
                inner = raw_predictions.get(key)
                if isinstance(inner, list):
                    raw_predictions = inner
                    break
            else:
                log_event(
                    logger, "llm_response_unexpected_shape", level=logging.WARNING,
                    top_level_keys=list(raw_predictions.keys()),
                    response_preview=text[:200],
                )
                raw_predictions = []

        if not isinstance(raw_predictions, list):
            log_event(logger, "llm_response_not_a_list", level=logging.WARNING,
                      chunk_size=len(chunk))
            return LLMClassifier._fallback_chunk(chunk, taxonomy, signals)

        # ---- Map predictions back onto the chunk ------------------------
        by_id = {p.get("transaction_id"): p for p in raw_predictions if isinstance(p, dict)}
        out = []
        for t in chunk:
            signal = signals.get(t.transaction_id)
            p = by_id.get(t.transaction_id)

            if p is None:
                out.append(_classify_via_plaid_or_heuristic(
                    t, valid_names, signal, taxonomy,
                    flow_if_plaid="llm_missing_plaid_fallback",
                    flow_if_heuristic="llm_missing_heuristic_fallback",
                    log_level=logging.INFO,
                ))
                continue

            predicted_class = str(p.get("predicted_class") or "")
            if predicted_class not in valid_names:
                out.append(_classify_via_plaid_or_heuristic(
                    t, valid_names, signal, taxonomy,
                    flow_if_plaid="llm_invalid_class_plaid_fallback",
                    flow_if_heuristic="llm_invalid_class_heuristic_fallback",
                    log_level=logging.WARNING,
                ))
                continue

            # Hard override: even if the LLM said REGULAR_PAYCHECK, don't
            # accept that specific claim when the amount/interval pattern
            # says this deposit doesn't match the counterparty's established
            # cadence. Only ever downgrades REGULAR_PAYCHECK, never anything
            # else.
            confidence = _parse_llm_confidence(p)
            if confidence is None:
                log_event(logger, "llm_invalid_confidence", level=logging.WARNING,
                          transaction_id=t.transaction_id,
                          raw_value=str(p.get("confidence"))[:40])
                out.append(_classify_via_plaid_or_heuristic(
                    t, valid_names, signal, taxonomy,
                    flow_if_plaid="llm_invalid_confidence_plaid_fallback",      # ← typo fixed
                    flow_if_heuristic="llm_invalid_confidence_heuristic_fallback",
                    log_level=logging.INFO,
                ))
                continue
            reason_code = _normalize_reason_code(p.get("reason_code"))
            is_guess = False
            flow = "llm"

            if (
                predicted_class == "REGULAR_PAYCHECK"
                and signal is not None
                and signal.has_enough_history
                and not signal.is_regular_candidate
            ):
                original_class = predicted_class
                predicted_class = "OTHER_W2" if "OTHER_W2" in valid_names else predicted_class
                confidence = min(confidence, 0.5)
                reason_code = "AMOUNT_INTERVAL_MISMATCH"
                is_guess = True
                flow = "llm_override_REGULAR_PAYCHECK"
                log_event(
                    logger, "REGULAR_PAYCHECK_override_applied", level=logging.INFO,
                    transaction_id=t.transaction_id,
                    llm_predicted_class=original_class,
                    overridden_to=predicted_class,
                    amount_ratio=signal.amount_ratio,
                    median_amount=signal.median_amount,
                    days_since_last=signal.days_since_last,
                    median_interval_days=signal.median_interval_days,
                    interval_deviation_days=signal.interval_deviation_days,
                )
            else:
                log_event(
                    logger, "transaction_classified", level=logging.DEBUG,
                    transaction_id=t.transaction_id, flow=flow,
                    predicted_class=predicted_class, confidence=confidence,
                    reason_code=reason_code,
                )

            alt = p.get("alternative_class")
            alt = str(alt) if alt in valid_names else None
            if alt is not None and alt == predicted_class:
                alt = None  # echoing the same class makes "second-best" meaningless
            out.append(
                _enrich(
                    t.transaction_id,
                    predicted_class,
                    confidence,
                    str(alt) if alt in valid_names else None,
                    reason_code,
                    direction=t.direction,
                    is_guess=is_guess,
                    taxonomy=taxonomy,
                    flow=flow,
                )
            )
        return out
