from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Optional, Literal
import re


@dataclass
class NormalizedTransaction:
    transaction_id: str
    account_type: str  # depository | credit | loan | investment | other
    amount: float  # Plaid convention: positive = money OUT, negative = money IN
    iso_currency_code: str
    date: str
    merchant_name: Optional[str]
    description: str
    plaid_category: list[str]
    plaid_category_detailed: Optional[str]
    is_recurring_hint: Optional[bool]
    counterparty_type: Optional[str]
    account_id: Optional[str] = None
    # Original, un-normalized transaction payload as received from the
    # backend/Plaid. Kept so the service layer can return the *complete*
    # transaction (all original fields) merged with classification output,
    # without the classifier/normalizer needing to know every possible
    # upstream schema.
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @property
    def direction(self) -> Literal["debit", "credit"]:
        """debit = money leaving the account (expense-like),
        credit = money entering the account (income-like)."""
        return "debit" if self.amount >= 0 else "credit"

    def to_dict(self) -> dict[str, Any]:
        """Compact dict used when talking to the LLM (excludes `raw` to save
        tokens and avoid leaking unrelated upstream fields into the prompt)."""
        d = asdict(self)
        d.pop("raw", None)
        d["direction"] = self.direction
        return d


_RECURRING_KEYWORDS = re.compile(
    r"\b(subscription|monthly|recurring|autopay|auto-pay|membership)\b", re.I
)
_P2P_MERCHANTS = re.compile(r"\b(venmo|zelle|cash app|cashapp|paypal)\b", re.I)


def _coerce_amount(raw_amount: Any, sign_convention: str) -> float:
    amount = float(raw_amount)
    if sign_convention == "flipped":
        amount = -amount
    return amount


def _infer_account_type(raw: dict[str, Any], account_meta: dict[str, Any]) -> str:
    if account_meta.get("type"):
        return str(account_meta["type"]).lower()
    if "account_type" in raw and raw["account_type"]:
        return str(raw["account_type"]).lower()
    if isinstance(raw.get("account"), dict):
        acct_type = raw["account"].get("type")
        if acct_type:
            return str(acct_type).lower()
    if raw.get("account_subtype") in ("credit card",):
        return "credit"
    return "depository"


def _infer_recurring(raw: dict[str, Any], description: str) -> Optional[bool]:
    if "is_recurring" in raw:
        return bool(raw["is_recurring"])
    if "stream_id" in raw or "recurring_transaction_id" in raw:
        return True
    if _RECURRING_KEYWORDS.search(description or ""):
        return True
    return None


def _infer_counterparty_type(raw: dict[str, Any], merchant: Optional[str], description: str) -> Optional[str]:
    if "counterparties" in raw and isinstance(raw["counterparties"], list) and raw["counterparties"]:
        cp = raw["counterparties"][0]
        if isinstance(cp, dict) and cp.get("type"):
            return str(cp["type"]).lower()
    text = f"{merchant or ''} {description or ''}"
    if _P2P_MERCHANTS.search(text):
        return "person"
    return None


def normalize_transaction(
    raw: dict[str, Any],
    *,
    account_meta: Optional[dict[str, Any]] = None,
    sign_convention: str = "standard",
) -> NormalizedTransaction:
    txn_id = raw.get("transaction_id") or raw.get("id") or raw.get("txn_id")
    if not txn_id:
        raise ValueError("Transaction is missing a usable id field")

    merchant_name = raw.get("merchant_name") or raw.get("merchant") or None
    description = raw.get("name") or raw.get("description") or raw.get("original_description") or ""

    # Safely resolve and cast plaid categories to list[str]
    pfc = raw.get("personal_finance_category")
    plaid_category: list[str] = []
    plaid_category_detailed: Optional[str] = None

    if isinstance(pfc, dict):
        detailed_val = pfc.get("detailed")
        plaid_category_detailed = str(detailed_val) if detailed_val is not None else None
        primary_val = pfc.get("primary")
        if primary_val is not None:
            plaid_category = [str(primary_val)]
    else:
        raw_category = raw.get("category")
        if isinstance(raw_category, list):
            plaid_category = [str(c) for c in raw_category if c is not None]
        detailed_cat = raw.get("category_detailed")
        plaid_category_detailed = str(detailed_cat) if detailed_cat is not None else None

    amount = _coerce_amount(raw.get("amount", 0.0), sign_convention)
    account_meta = account_meta or {}

    currency = (
        raw.get("iso_currency_code")
        or raw.get("currency")
        or account_meta.get("iso_currency_code")
        or "USD"
    )
    date_val = raw.get("date") or raw.get("authorized_date") or raw.get("posted_date") or ""
    account_id = raw.get("account_id")

    return NormalizedTransaction(
        transaction_id=str(txn_id),
        account_type=_infer_account_type(raw, account_meta),
        amount=amount,
        iso_currency_code=str(currency),
        date=str(date_val),
        merchant_name=str(merchant_name) if merchant_name is not None else None,
        description=str(description),
        plaid_category=plaid_category,
        plaid_category_detailed=plaid_category_detailed,
        is_recurring_hint=_infer_recurring(raw, str(description)),
        counterparty_type=_infer_counterparty_type(raw, merchant_name, str(description)),
        account_id=str(account_id) if account_id is not None else None,
        raw=raw,
    )


def normalize_plaid_response(
    payload: dict[str, Any] | list[Any],
    *,
    sign_convention: str = "standard",
) -> list[NormalizedTransaction]:
    accounts_map: dict[str, dict[str, Any]] = {}
    transactions_list: list[Any] = []

    if isinstance(payload, dict):
        accounts_data = payload.get("accounts", [])
        if isinstance(accounts_data, list):
            for acct in accounts_data:
                if isinstance(acct, dict) and "account_id" in acct:
                    acct_id_val = acct["account_id"]
                    if acct_id_val is not None:
                        accounts_map[str(acct_id_val)] = acct

        added_data = payload.get("added", [])
        if isinstance(added_data, list) and added_data:
            transactions_list = added_data
        elif "transactions" in payload and isinstance(payload["transactions"], list):
            transactions_list = payload["transactions"]
    elif isinstance(payload, list):
        transactions_list = payload

    out: list[NormalizedTransaction] = []
    for raw in transactions_list:
        try:
            if not isinstance(raw, dict):
                continue

            # Guard acct_id lookup against None / unknown types
            acct_id = raw.get("account_id")
            account_meta = {}
            if acct_id is not None:
                account_meta = accounts_map.get(str(acct_id), {})

            out.append(normalize_transaction(raw, account_meta=account_meta, sign_convention=sign_convention))
        except (ValueError, TypeError, KeyError) as exc:
            print(f"[normalizer] skipped unparsable transaction: {exc}")

    return out
