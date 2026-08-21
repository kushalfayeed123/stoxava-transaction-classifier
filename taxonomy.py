"""
Client-defined class taxonomy.

In production this would be provided by the client (per the plan) and could
be loaded from a config file / DB table instead of hardcoded here. This
default set is a placeholder covering the classes requested for the demo
(Salaries, Groceries, Gifts, Subscriptions, Dining) plus a reasonable set of
common personal-finance classes so the demo doesn't look thin, and an
"Uncategorized" catch-all that always exists regardless of what the client
provides.

Each class carries a `flow` hint used to derive the transaction-level
income/expense/transfer flag:

- "income":   this class is virtually always money coming in (Salary).
- "expense":  this class is virtually always money going out (Dining, ...).
- "transfer": movement between accounts/people that isn't really income or
              expense (Transfers, ATM/Cash).
- "ambiguous": the class itself doesn't determine direction (e.g. "Gifts"
              can be sent or received) -- the actual amount sign on the
              transaction is used to resolve it at classification time.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

Flow = Literal["income", "expense", "transfer", "ambiguous"]


@dataclass(frozen=True)
class TxnClass:
    name: str
    definition: str
    flow: Flow = "expense"


DEFAULT_TAXONOMY: list[TxnClass] = [
    TxnClass("Salary", "Payroll deposits, direct deposit from an employer, wages.", "income"),
    TxnClass("Groceries", "Supermarkets, grocery stores, food markets.", "expense"),
    TxnClass("Gifts", "Peer-to-peer transfers to individuals for gifts/personal reasons (Venmo, Zelle, Cash App to a person).", "ambiguous"),
    TxnClass("Subscriptions", "Recurring fixed-amount charges for digital or membership services (streaming, software, gym).", "expense"),
    TxnClass("Dining", "Restaurants, cafes, bars, food delivery.", "expense"),
    TxnClass("Rent/Mortgage", "Recurring housing payments to a landlord, property manager, or mortgage servicer.", "expense"),
    TxnClass("Utilities", "Electricity, gas, water, internet, phone bills.", "expense"),
    TxnClass("Transportation", "Ride-share, public transit, fuel, parking, vehicle maintenance.", "expense"),
    TxnClass("Shopping", "General retail purchases, e-commerce, clothing, electronics.", "expense"),
    TxnClass("Entertainment", "Movies, events, concerts, hobbies, games (non-subscription, one-off spend).", "expense"),
    TxnClass("Healthcare", "Pharmacies, clinics, hospitals, medical/dental/vision providers, insurance copays.", "expense"),
    TxnClass("Travel", "Airlines, hotels, car rentals, travel booking platforms.", "expense"),
    TxnClass("Transfers", "Movement of funds between the user's own accounts, or non-gift P2P transfers.", "transfer"),
    TxnClass("ATM/Cash", "Cash withdrawals.", "transfer"),
    TxnClass("Loan Payment", "Payments toward student loans, personal loans, or auto loans (not mortgage).", "expense"),
    TxnClass("Insurance", "Auto, home/renters, or life insurance premium payments.", "expense"),
    TxnClass("Fees/Charges", "Bank fees, interest charges, overdraft fees, service charges.", "expense"),
    TxnClass("Refund/Reimbursement", "Money returned to the account: merchant refunds, chargebacks, reimbursements.", "income"),
    TxnClass("Investment", "Brokerage transfers, buys/sells, retirement contributions.", "ambiguous"),
    TxnClass("Uncategorized", "Does not clearly fit any other class, or insufficient information to classify.", "ambiguous"),
]

TAXONOMY_BY_NAME: dict[str, TxnClass] = {c.name: c for c in DEFAULT_TAXONOMY}


def taxonomy_to_prompt_block(taxonomy: list[TxnClass]) -> str:
    """Render the taxonomy as the AVAILABLE_CLASSES block injected into the
    user message sent alongside the system prompt."""
    lines = [f'- "{c.name}": {c.definition}' for c in taxonomy]
    return "AVAILABLE_CLASSES:\n" + "\n".join(lines)


def resolve_flow_type(class_name: str, direction: Literal["debit", "credit"], taxonomy: list[TxnClass] | None = None) -> str:
    """Resolve the final income/expense/transfer label for a transaction.

    Combines the class's semantic flow with the transaction's actual amount
    direction, so that "ambiguous" classes (Gifts, Investment, Uncategorized)
    still get a sensible income/expense label a backend can filter on.
    """
    by_name = TAXONOMY_BY_NAME if taxonomy is None else {c.name: c for c in taxonomy}
    cls = by_name.get(class_name)
    flow = cls.flow if cls else "ambiguous"

    if flow in ("income", "expense", "transfer"):
        return flow
    # ambiguous -> resolve from direction (credit = money in = income-like)
    return "income" if direction == "credit" else "expense"
