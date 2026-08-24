"""
Classification backends.

LLMClassifier calls a Gemini model using the system prompt in
system_prompt.md via the google-genai SDK -- requires GEMINI_API_KEY.

Every call returns the `Prediction` dataclass so the UI and evaluator never
need to know how a result was produced.

Design principle: **every transaction always gets a predicted_class.**
Nothing is ever left unclassified. When there truly isn't enough signal to
make a confident call, the classifier falls back to `_heuristic_guess`,
which makes the best possible guess from direction / recurrence /
counterparty signals and reports a low (but non-zero, non-arbitrary)
confidence score plus `is_guess=True` so downstream consumers can tell
"classified" apart from "guessed."
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from normalizer import NormalizedTransaction
from taxonomy import TxnClass, taxonomy_to_prompt_block, resolve_flow_type, \
    Direction

SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent / "system_prompt.md"

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


def _match_plaid_category(t: NormalizedTransaction, valid_names: set[str]) -> Optional[tuple[str, float, str]]:
    """Returns (predicted_class, confidence, reason_code) from Plaid's own
    category fields, or None if nothing usable is present."""
    detailed = (t.plaid_category_detailed or "").upper()
    if detailed in _PFC_DETAILED_MAP:
        cls = _PFC_DETAILED_MAP[detailed]
        if cls in valid_names:
            return cls, 0.78, "PLAID_CATEGORY_MATCH"

    for token in t.plaid_category:
        primary = token.upper().replace(" ", "_")
        if primary in _PFC_PRIMARY_MAP:
            cls = _PFC_PRIMARY_MAP[primary]
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

    NEEDS_REVIEW_THRESHOLD = 0.6

    @property
    def needs_review(self) -> bool:
        return self.confidence < self.NEEDS_REVIEW_THRESHOLD


class Classifier:
    """Interface implemented by the LLM backend."""

    def classify_batch(
        self,
        transactions: list[NormalizedTransaction],
        taxonomy: list[TxnClass],
    ) -> list[Prediction]:
        raise NotImplementedError

# --------------------------------------------------------------------------
# Shared fallback: never leave a transaction with no guess at all.
# --------------------------------------------------------------------------


def _heuristic_guess(t: NormalizedTransaction, valid_names: set[str]) -> tuple[str, float, str]:
    text = f"{t.merchant_name or ''} {t.description}".lower()

    if t.direction == "credit":
        # Strong textual signal only -- never guess a specific income type
        # without evidence, because income treatment (gross projection,
        # confirmation requirements) differs per class.
        if any(k in text for k in ("payroll", "salary", "direct dep", "paycheck")) \
                and "REGULAR_PAYCHECK" in valid_names:
            return "REGULAR_PAYCHECK", 0.6, "MERCHANT_MATCH"
        if t.is_recurring_hint and "REGULAR_PAYCHECK" in valid_names:
            return "REGULAR_PAYCHECK", 0.55, "RECURRING_PATTERN"
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
    )

# --------------------------------------------------------------------------
# LLM-backed classifier (production path using Gemini Free Tier)
# --------------------------------------------------------------------------


class LLMClassifier(Classifier):

    def __init__(self, model: str="gemini-2.5-flash", batch_size: int=25):
        try:
            from google import genai
            from google.genai import types  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "The 'google-genai' package is required for LLMClassifier. "
                "Install with: uv add google-genai"
            ) from exc

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set.")
        if not SYSTEM_PROMPT_PATH.exists():
            raise FileNotFoundError(
                f"system_prompt.md not found at {SYSTEM_PROMPT_PATH}. "
                "LLMClassifier requires this file alongside classifier.py."
            )
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._batch_size = batch_size
        self._system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")

    def classify_batch(
        self,
        transactions: list[NormalizedTransaction],
        taxonomy: list[TxnClass],
    ) -> list[Prediction]:
        results: list[Prediction] = []
        for start in range(0, len(transactions), self._batch_size):
            chunk = transactions[start: start + self._batch_size]
            try:
                results.extend(self._classify_chunk(chunk, taxonomy))
            except Exception:
                # Never let a transient API/parsing failure take the whole
                # batch down -- fall back to heuristic guesses for this chunk.
                results.extend(self._fallback_chunk(chunk, taxonomy))
        return results

    @staticmethod
    def _fallback_chunk(chunk: list[NormalizedTransaction], taxonomy: list[TxnClass]) -> list[Prediction]:
        valid_names = {c.name for c in taxonomy}
        return [
            _enrich(
                t.transaction_id,
                *_match_plaid_category(t, valid_names) or _heuristic_guess(t, valid_names),
                direction=t.direction,
                is_guess=True,
                taxonomy=taxonomy,
            )
            for t in chunk
        ]

    def _classify_chunk(
        self, chunk: list[NormalizedTransaction], taxonomy: list[TxnClass]
    ) -> list[Prediction]:
        from google.genai import types

        example = json.dumps(
            [{
                "transaction_id": chunk[0].transaction_id,
                "predicted_class": "<class name from taxonomy>",
                "confidence": 0.9,
                "alternative_class": None,
                "reason_code": "MERCHANT_MATCH",
            }],
            indent=2,
        )

        user_content = (
            taxonomy_to_prompt_block(taxonomy)
            +"Respond with ONLY a JSON array of objects matching this shape:" + example + "TRANSACTIONS:"
            +json.dumps([t.to_dict() for t in chunk], indent=2)
        )

        response = self._client.models.generate_content(
            model=self._model,
            contents=user_content,
            config=types.GenerateContentConfig(
                system_instruction=self._system_prompt,
                temperature=0.0,
                max_output_tokens=4000,
                response_mime_type="application/json",
            ),
        )

        text = response.text or ""
        return self._parse_response(text, chunk, taxonomy)

    @staticmethod
    def _parse_response(
        text: str, chunk: list[NormalizedTransaction], taxonomy: list[TxnClass]
    ) -> list[Prediction]:
        valid_names = {c.name for c in taxonomy}

        cleaned = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
        try:
            raw_predictions = json.loads(cleaned)
        except json.JSONDecodeError:
            return LLMClassifier._fallback_chunk(chunk, taxonomy)

        if isinstance(raw_predictions, dict):
            for key in ("transactions", "predictions", "results", "data"):
                inner = raw_predictions.get(key)
                if isinstance(inner, list):
                    raw_predictions = inner
                    break
            else:
                raw_predictions = []

        if not isinstance(raw_predictions, list):
            return LLMClassifier._fallback_chunk(chunk, taxonomy)

        by_id = {p.get("transaction_id"): p for p in raw_predictions if isinstance(p, dict)}
        out = []
        for t in chunk:
            p = by_id.get(t.transaction_id)
            plaid_match = _match_plaid_category(t, valid_names)

            if p is None:
                cls, conf, reason = plaid_match or _heuristic_guess(t, valid_names)
                out.append(_enrich(t.transaction_id, cls, conf, None, reason,
                                   direction=t.direction, is_guess=True, taxonomy=taxonomy))
                continue

            predicted_class = str(p.get("predicted_class") or "")
            if predicted_class not in valid_names:
                cls, conf, reason = plaid_match or _heuristic_guess(t, valid_names)
                out.append(_enrich(t.transaction_id, cls, conf, None, reason,
                                   direction=t.direction, is_guess=True, taxonomy=taxonomy))
                continue

            out.append(
                _enrich(
                    t.transaction_id,
                    predicted_class,
                    float(p.get("confidence", 0.0)),
                    p.get("alternative_class"),
                    str(p.get("reason_code", "AMBIGUOUS")),
                    direction=t.direction,
                    is_guess=False,
                    taxonomy=taxonomy,
                )
            )
        return out
