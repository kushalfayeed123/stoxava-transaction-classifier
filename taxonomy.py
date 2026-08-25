"""
Client-defined class taxonomy -- STOXAVA two-taxonomy model.

STOXAVA classifies against two *separate* taxonomies depending on which way
money is moving (per STOXAVA_Expense_Classification_Document and
STOXAVA_Income_Page_V5_Detailed):

- direction == "debit"  (money OUT) -> EXPENSE_TAXONOMY (18 categories,
  each with a default top-level Expense Plan bucket).
- direction == "credit" (money IN)  -> INCOME_TAXONOMY (15 deposit
  categories, each with actual/projected-gross treatment flags).

Both taxonomies are JSON-backed (see `taxonomy_data/`) so a client can hand
STOXAVA a new taxonomy file -- add a category, change a bucket, flip a
treatment flag -- without a code change. This module supplies the default
JSON (written to disk on first import if missing), the dataclass shape, and
the lookup helpers the rest of the pipeline (classifier/service/evaluator)
depends on.

Backward-compat note: `DEFAULT_TAXONOMY` still exists as the union of both
taxonomies, for any caller that hasn't been updated to pick a taxonomy by
direction yet (see `taxonomy_for_direction`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal, Optional

Flow = Literal["income", "expense", "transfer", "ambiguous"]
Direction = Literal["debit", "credit"]

TAXONOMY_DATA_DIR = Path(__file__).resolve().parent / "taxonomy_data"
EXPENSE_TAXONOMY_PATH = TAXONOMY_DATA_DIR / "expense_taxonomy.json"
INCOME_TAXONOMY_PATH = TAXONOMY_DATA_DIR / "income_taxonomy.json"

# Top-level Expense Plan buckets (STOXAVA_Expense_Classification_Document,
# section 3). Order matters for display; "Needs Review" and
# "Excluded / Transfers" are not spend buckets, they're routing states.
TOP_LEVEL_BUCKETS: list[str] = [
    "Monthly Living Expenses",
    "Debt Payments",
    "Insurance",
    "Family Support",
    "Future Expense Savings",
    "Flexible Spending",
    "Planning Room",
    "Excluded / Transfers",
    "Needs Review",
]


@dataclass(frozen=True)
class TxnClass:
    """One category within a direction-specific taxonomy.

    Expense-side fields (`top_level_bucket`, `needs_review_default`) are set
    for classes in EXPENSE_TAXONOMY. Income-side fields (`counts_actual_income`,
    `counts_projected_gross`, `requires_gross_confirmation`) are set for
    classes in INCOME_TAXONOMY. A class only ever belongs to one taxonomy,
    so the fields that don't apply are left at their defaults (None/False).
    """

    name: str
    definition: str
    flow: Flow = "expense"
    direction: Direction = "debit"  # which taxonomy this class lives in

    # -- expense-taxonomy fields --
    top_level_bucket: Optional[str] = None
    needs_review_default: bool = False  # this class itself is ambiguous
                                         # (e.g. "Needs Review", "Amazon"-like)

    # -- income-taxonomy fields (per Income doc section 14 treatment table) --
    counts_actual_income: Optional[bool] = None
    # "no" | "yes" | "yes_once_gross_confirmed" -- kept as a string rather
    # than a bool because the "once confirmed" case is neither a flat yes
    # nor a flat no.
    counts_projected_gross: Optional[Literal["no", "yes", "yes_once_gross_confirmed"]] = None
    requires_gross_confirmation: Optional[bool] = None

    def to_json(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# Default taxonomies (used to seed taxonomy_data/*.json on first run, and as
# the in-memory fallback if the JSON files are ever missing/corrupt).
# --------------------------------------------------------------------------

_DEFAULT_EXPENSE_TAXONOMY: list[TxnClass] = [
    TxnClass("Housing", "Rent, mortgage payments on a primary home, property management.",
             "expense", "debit", top_level_bucket="Monthly Living Expenses"),
    TxnClass("Groceries", "Supermarkets, grocery stores, food markets.",
             "expense", "debit", top_level_bucket="Monthly Living Expenses"),
    TxnClass("Dining", "Restaurants, cafes, bars, food delivery.",
             "expense", "debit", top_level_bucket="Monthly Living Expenses"),
    TxnClass("Transportation", "Ride-share, public transit, fuel, parking, vehicle maintenance.",
             "expense", "debit", top_level_bucket="Monthly Living Expenses"),
    TxnClass("Travel", "Airlines, hotels, car rentals, travel booking platforms.",
             "expense", "debit", top_level_bucket="Monthly Living Expenses"),
    TxnClass("Utilities", "Electricity, gas, water, internet, phone bills.",
             "expense", "debit", top_level_bucket="Monthly Living Expenses"),
    TxnClass("Subscriptions", "Recurring fixed-amount charges for digital or membership services (streaming, software, gym).",
             "expense", "debit", top_level_bucket="Monthly Living Expenses"),
    TxnClass("Shopping", "General retail purchases, e-commerce, clothing, electronics.",
             "expense", "debit", top_level_bucket="Monthly Living Expenses"),
    TxnClass("Healthcare", "Pharmacies, clinics, hospitals, medical/dental/vision providers, insurance copays.",
             "expense", "debit", top_level_bucket="Monthly Living Expenses"),
    TxnClass("Education", "Tuition, school fees, courses, exam/certification fees.",
             "expense", "debit", top_level_bucket="Monthly Living Expenses"),
    TxnClass("Family Support", "Recurring or confirmed transfers supporting family members (rent, remittances, dependent care).",
             "expense", "debit", top_level_bucket="Family Support"),
    TxnClass("Debt Payments", "Student loan, personal loan, auto loan, and credit-card minimum/partial payments.",
             "expense", "debit", top_level_bucket="Debt Payments"),
    TxnClass("Insurance", "Auto, home/renters, health (outside payroll), and life insurance premium payments.",
             "expense", "debit", top_level_bucket="Insurance"),
    TxnClass("Personal Care", "Salons, gyms (non-membership), grooming, cosmetics, personal wellness.",
             "expense", "debit", top_level_bucket="Monthly Living Expenses"),
    TxnClass("Entertainment", "Movies, events, concerts, hobbies, games (non-subscription, one-off spend).",
             "expense", "debit", top_level_bucket="Monthly Living Expenses"),
    TxnClass("Other", "Does not clearly fit any other expense class but is still confidently an expense.",
             "expense", "debit", top_level_bucket="Monthly Living Expenses"),
    TxnClass("Transfers / Excluded", "Movement between the user's own accounts, full credit-card payments, savings/brokerage/retirement/HSA contributions, CD deposits -- excluded from the Expense Plan to avoid double counting.",
             "transfer", "debit", top_level_bucket="Excluded / Transfers"),
    TxnClass("Needs Review", "Could plausibly belong to more than one category (Amazon, Walmart, Costco, P2P apps, large one-time purchases, unknown merchants) -- routed to the user before it is counted anywhere.",
             "ambiguous", "debit", top_level_bucket="Needs Review", needs_review_default=True),
]

# Income / deposit taxonomy (Income doc section 14). Treatment columns come
# directly from the doc's table for the 12 classes it names explicitly.
# OTHER_W2 is grouped with the doc's "variable W2 income" bucket (section 3B)
# so it gets the same treatment as Bonus/Commission/etc. OTHER_INCOME and
# FOREIGN_SOURCE aren't in the treatment table; they're modeled as
# non-W2, review-first income the same way UNCLASSIFIED is, since the doc's
# core rule is "never invent gross income STOXAVA can't defend." These three
# are the one place this taxonomy extrapolates beyond the doc -- flag that
# to the client if the intended treatment differs.
_DEFAULT_INCOME_TAXONOMY: list[TxnClass] = [
    TxnClass("REGULAR_W2", "Recurring payroll/direct-deposit paycheck from an employer.",
             "income", "credit",
             counts_actual_income=True, counts_projected_gross="no", requires_gross_confirmation=False),
    TxnClass("BONUS", "One-time or periodic bonus payment from an employer.",
             "income", "credit",
             counts_actual_income=True, counts_projected_gross="yes_once_gross_confirmed", requires_gross_confirmation=True),
    TxnClass("COMMISSION", "Sales commission payment from an employer.",
             "income", "credit",
             counts_actual_income=True, counts_projected_gross="yes_once_gross_confirmed", requires_gross_confirmation=True),
    TxnClass("OVERTIME", "Overtime pay, typically paid alongside or separate from a regular paycheck.",
             "income", "credit",
             counts_actual_income=True, counts_projected_gross="yes_once_gross_confirmed", requires_gross_confirmation=True),
    TxnClass("RETRO_PAY", "Retroactive pay correction from an employer.",
             "income", "credit",
             counts_actual_income=True, counts_projected_gross="yes_once_gross_confirmed", requires_gross_confirmation=True),
    TxnClass("SEVERANCE", "Severance payment from a former employer.",
             "income", "credit",
             counts_actual_income=True, counts_projected_gross="yes_once_gross_confirmed", requires_gross_confirmation=True),
    TxnClass("OTHER_W2", "Other W-2 employer compensation not covered by the classes above.",
             "income", "credit",
             counts_actual_income=True, counts_projected_gross="yes_once_gross_confirmed", requires_gross_confirmation=True),
    TxnClass("REIMBURSEMENT", "Employer or third-party reimbursement of an out-of-pocket expense.",
             "transfer", "credit",
             counts_actual_income=False, counts_projected_gross="no", requires_gross_confirmation=False),
    TxnClass("TRANSFER", "Movement of funds between the user's own accounts, or a non-gift P2P transfer.",
             "transfer", "credit",
             counts_actual_income=False, counts_projected_gross="no", requires_gross_confirmation=False),
    TxnClass("REFUND", "Merchant refund or chargeback credited back to the account.",
             "transfer", "credit",
             counts_actual_income=False, counts_projected_gross="no", requires_gross_confirmation=False),
    TxnClass("LOAN", "Loan proceeds or a cash-advance deposit.",
             "transfer", "credit",
             counts_actual_income=False, counts_projected_gross="no", requires_gross_confirmation=False),
    TxnClass("GIFT_FAMILY_SUPPORT", "P2P deposit from an individual for gift or family-support purposes -- not W-2 income.",
             "ambiguous", "credit",
             counts_actual_income=False, counts_projected_gross="no", requires_gross_confirmation=False),
    TxnClass("OTHER_INCOME", "Confirmed non-W2 income that doesn't fit another class (not yet reflected in gross-income projections).",
             "ambiguous", "credit",
             counts_actual_income=True, counts_projected_gross="no", requires_gross_confirmation=False),
    TxnClass("FOREIGN_SOURCE", "Deposit from a foreign source; treated as review-first until classified.",
             "ambiguous", "credit",
             counts_actual_income=False, counts_projected_gross="no", requires_gross_confirmation=False,
             needs_review_default=True),
    TxnClass("UNCLASSIFIED", "Insufficient information to classify the deposit -- excluded from income until reviewed.",
             "ambiguous", "credit",
             counts_actual_income=False, counts_projected_gross="no", requires_gross_confirmation=False,
             needs_review_default=True),
]


# --------------------------------------------------------------------------
# JSON persistence
# --------------------------------------------------------------------------

def _write_default_json(path: Path, classes: list[TxnClass]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([c.to_json() for c in classes], indent=2), encoding="utf-8")


def _load_json(path: Path) -> Optional[list[TxnClass]]:
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return [TxnClass(**item) for item in raw]
    except (json.JSONDecodeError, TypeError, OSError) as exc:
        print(f"[taxonomy] failed to load {path}, falling back to built-in defaults: {exc}")
        return None


def load_taxonomy(direction: Direction) -> list[TxnClass]:
    """Load the taxonomy for a given direction from taxonomy_data/*.json,
    seeding the file with the built-in defaults on first run."""
    path = EXPENSE_TAXONOMY_PATH if direction == "debit" else INCOME_TAXONOMY_PATH
    default = _DEFAULT_EXPENSE_TAXONOMY if direction == "debit" else _DEFAULT_INCOME_TAXONOMY

    loaded = _load_json(path)
    if loaded is not None:
        return loaded

    _write_default_json(path, default)
    return default


def save_taxonomy(direction: Direction, classes: list[TxnClass]) -> None:
    """Persist a (possibly client-edited) taxonomy back to disk."""
    path = EXPENSE_TAXONOMY_PATH if direction == "debit" else INCOME_TAXONOMY_PATH
    _write_default_json(path, classes)


EXPENSE_TAXONOMY: list[TxnClass] = load_taxonomy("debit")
INCOME_TAXONOMY: list[TxnClass] = load_taxonomy("credit")

# Backward-compat: union of both, for callers not yet updated to select a
# taxonomy by direction.
DEFAULT_TAXONOMY: list[TxnClass] = EXPENSE_TAXONOMY + INCOME_TAXONOMY

TAXONOMY_BY_NAME: dict[str, TxnClass] = {c.name: c for c in DEFAULT_TAXONOMY}


def taxonomy_for_direction(direction: Direction) -> list[TxnClass]:
    """The taxonomy STOXAVA classifies against for a given txn direction:
    expense categories for money OUT, income/deposit categories for money IN."""
    return EXPENSE_TAXONOMY if direction == "debit" else INCOME_TAXONOMY


def taxonomy_to_prompt_block(taxonomy: list[TxnClass]) -> str:
    """Render a taxonomy as the AVAILABLE_CLASSES block injected into the
    user message sent alongside the system prompt."""
    lines = [f'- "{c.name}": {c.definition}' for c in taxonomy]
    return "AVAILABLE_CLASSES:\n" + "\n".join(lines)


def resolve_flow_type(class_name: str, direction: Direction, taxonomy: list[TxnClass] | None = None) -> str:
    """Resolve the final income/expense/transfer label for a transaction.

    Combines the class's semantic flow with the transaction's actual amount
    direction, so that "ambiguous" classes (GIFT_FAMILY_SUPPORT, Investment,
    Needs Review, ...) still get a sensible income/expense label a backend
    can filter on.
    """
    by_name = TAXONOMY_BY_NAME if taxonomy is None else {c.name: c for c in taxonomy}
    cls = by_name.get(class_name)
    flow = cls.flow if cls else "ambiguous"

    if flow in ("income", "expense", "transfer"):
        return flow
    # ambiguous -> resolve from direction (credit = money in = income-like)
    return "income" if direction == "credit" else "expense"


def default_top_level_bucket(class_name: str, taxonomy: list[TxnClass] | None = None) -> Optional[str]:
    """The default Expense Plan bucket for an expense-side class, or None
    for income-side classes (which don't have buckets)."""
    by_name = TAXONOMY_BY_NAME if taxonomy is None else {c.name: c for c in taxonomy}
    cls = by_name.get(class_name)
    return cls.top_level_bucket if cls else None


def income_treatment(class_name: str, taxonomy: list[TxnClass] | None = None) -> dict:
    """The actual/projected-gross treatment for an income-side class, or
    all-False/None if the class isn't an income class (not applicable)."""
    by_name = TAXONOMY_BY_NAME if taxonomy is None else {c.name: c for c in taxonomy}
    cls = by_name.get(class_name)
    if cls is None or cls.counts_actual_income is None:
        return {
            "counts_actual_income": False,
            "counts_projected_gross": "no",
            "requires_gross_confirmation": False,
        }
    return {
        "counts_actual_income": cls.counts_actual_income,
        "counts_projected_gross": cls.counts_projected_gross,
        "requires_gross_confirmation": cls.requires_gross_confirmation,
    }