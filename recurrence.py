"""
Amount + timing recurrence signals for income (credit) transactions.

Problem this solves: a classifier working transaction-by-transaction has no
way to know that "$1,850 from ADP every 2 weeks" is the user's real paycheck
pattern, so it has no basis to reject a $310 deposit from the same ADP
counterparty that shows up 4 days after the last "real" paycheck. Text/
recurring-hint signals alone will happily call that REGULAR_W2 too, which is
wrong -- a paycheck that's the wrong amount and not on the usual schedule
cannot be a paycheck, it's something else (partial pay, correction, a
different W2 line item, or unrelated).

This module groups an account's credit transactions by normalized
counterparty, learns each group's median amount and median interval between
deposits, and reports how much every transaction in that group deviates from
its own group's norm. `is_regular_candidate` is the hard gate:
REGULAR_W2 should only ever be assigned when this is True (or when there's
not yet enough history to judge, i.e. group_size < 2).

IMPORTANT SCOPE NOTE: signals are computed only from transactions passed
into `compute_recurrence_signals` in a single call. If classify_batch is
invoked per-sync (e.g. only "added" transactions from an incremental Plaid
sync) rather than with the account's full transaction history, the group
stats will be based on a partial window and less reliable early on. For best
accuracy, call this against the account's full known transaction history,
not just a single incremental batch.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from datetime import date as date_type, datetime
from typing import Optional

from normalizer import NormalizedTransaction

# Tolerance knobs -- tune against real client data once you have enough
# paycheck history to see what "normal" variance looks like (e.g. paychecks
# that vary a few dollars due to changed tax withholding shouldn't trip this).
AMOUNT_TOLERANCE = 0.15        # +/-15% of the group's median amount
INTERVAL_TOLERANCE_DAYS = 5    # +/-5 days off the group's median cadence

_NUMERIC_RUN = re.compile(r"\b\d{3,}\b")
_NON_ALNUM = re.compile(r"[^a-z0-9 ]")


def _normalize_counterparty(t: NormalizedTransaction) -> str:
    """Collapse merchant/description text to something stable enough to
    group repeat deposits from "the same place" (strips trailing reference
    numbers, store numbers, punctuation)."""
    name = (t.merchant_name or t.description or "").lower()
    name = _NON_ALNUM.sub(" ", name)
    name = _NUMERIC_RUN.sub(" ", name)
    return re.sub(r"\s+", " ", name).strip()


def _parse_date(t: NormalizedTransaction) -> Optional[date_type]:
    if not t.date:
        return None
    try:
        return datetime.fromisoformat(t.date[:10]).date()
    except ValueError:
        return None


@dataclass
class RecurrenceSignal:
    group_key: str
    group_size: int  # total dated deposits seen from this counterparty (this txn included)
    median_amount: Optional[float]         # group's median deposit magnitude (abs value)
    amount_ratio: Optional[float]          # this txn's |amount| / median_amount
    median_interval_days: Optional[float]  # group's typical days-between-deposits
    days_since_last: Optional[int]         # days since the previous deposit from this group
    interval_deviation_days: Optional[float]
    is_regular_candidate: bool             # amount AND interval both within tolerance

    @property
    def has_enough_history(self) -> bool:
        return self.group_size >= 2

    def to_prompt_fragment(self) -> str:
        """Short human-readable line to inject alongside this transaction in
        the LLM prompt, so the model gets the same signal the code enforces."""
        if not self.has_enough_history:
            return "no prior deposit history from this counterparty yet"
        bits = []
        if self.amount_ratio is not None:
            bits.append(f"amount is {self.amount_ratio:.0%} of this counterparty's usual deposit")
        if self.interval_deviation_days is not None:
            bits.append(
                f"{self.days_since_last}d since the last one "
                f"(usual ~{self.median_interval_days:.0f}d, off by {self.interval_deviation_days:.0f}d)"
            )
        verdict = "MATCHES the regular pattern" if self.is_regular_candidate else "DOES NOT match the regular pattern"
        return "; ".join(bits) + f" -> {verdict}"


_NEUTRAL_SIGNAL_KWARGS = dict(
    median_amount=None, amount_ratio=None, median_interval_days=None,
    days_since_last=None, interval_deviation_days=None, is_regular_candidate=False,
)


def compute_recurrence_signals(
    transactions: list[NormalizedTransaction],
) -> dict[str, RecurrenceSignal]:
    """Returns transaction_id -> RecurrenceSignal for every credit
    transaction in `transactions`. Debit transactions are ignored (they're
    not passed in classify_batch's income path anyway, but this is
    defensive so callers can pass a mixed list safely)."""
    credits = [t for t in transactions if t.direction == "credit"]

    groups: dict[tuple[Optional[str], str], list[tuple[NormalizedTransaction, Optional[date_type]]]] = {}
    for t in credits:
        # Key by account too -- the same merchant name shouldn't blend
        # histories across a user's different accounts.
        key = (t.account_id, _normalize_counterparty(t))
        groups.setdefault(key, []).append((t, _parse_date(t)))

    signals: dict[str, RecurrenceSignal] = {}

    for (account_id, name_key), items in groups.items():
        undated = [ti for ti in items if ti[1] is None]
        dated = sorted((ti for ti in items if ti[1] is not None), key=lambda ti: ti[1])

        amounts = [abs(ti[0].amount) for ti in dated]
        median_amount = statistics.median(amounts) if len(amounts) >= 2 else None

        intervals = [(dated[i][1] - dated[i - 1][1]).days for i in range(1, len(dated))]
        median_interval = statistics.median(intervals) if intervals else None

        for i, (t, d) in enumerate(dated):
            prev_date = dated[i - 1][1] if i > 0 else None
            days_since_last = (d - prev_date).days if prev_date else None

            amount_ratio = (abs(t.amount) / median_amount) if median_amount else None
            interval_deviation = (
                abs(days_since_last - median_interval)
                if (days_since_last is not None and median_interval is not None)
                else None
            )

            is_regular = (
                amount_ratio is not None
                and interval_deviation is not None
                and (1 - AMOUNT_TOLERANCE) <= amount_ratio <= (1 + AMOUNT_TOLERANCE)
                and interval_deviation <= INTERVAL_TOLERANCE_DAYS
            )

            signals[t.transaction_id] = RecurrenceSignal(
                group_key=f"{account_id or '?'}:{name_key}",
                group_size=len(dated),
                median_amount=median_amount,
                amount_ratio=amount_ratio,
                median_interval_days=median_interval,
                days_since_last=days_since_last,
                interval_deviation_days=interval_deviation,
                is_regular_candidate=is_regular,
            )

        # Undated transactions (bad/missing date) can't be judged for
        # timing -- give them a neutral, non-regular signal rather than
        # dropping them, so every credit txn always has a lookup entry.
        for t, _ in undated:
            signals[t.transaction_id] = RecurrenceSignal(
                group_key=f"{account_id or '?'}:{name_key}",
                group_size=len(items),
                **_NEUTRAL_SIGNAL_KWARGS,
            )

    return signals
