"""
End-to-end offline demo: dummy_data -> service.classify_plaid_response
(MockClassifier) -> evaluator.

Run: python demo.py
"""
from __future__ import annotations

import json

from classifier import MockClassifier, Prediction
from dummy_data import get_dummy_transactions_with_labels
from evaluator import evaluate, print_report
from service import classify_plaid_response
from taxonomy import DEFAULT_TAXONOMY


def main() -> None:
    raw_transactions, true_labels = get_dummy_transactions_with_labels()
    classifier = MockClassifier()

    result = classify_plaid_response(raw_transactions, classifier, taxonomy=DEFAULT_TAXONOMY)

    print(f"Backend: {result['backend']}")
    print(f"Summary: {json.dumps(result['summary'], indent=2)}")
    print(f"\nSample classified transaction:")
    print(json.dumps(result["transactions"][0], indent=2, default=str))

    # Rebuild Prediction objects from the merged output so we can reuse the
    # existing evaluator without changes.
    predictions = [
        Prediction(
            transaction_id=t.get("transaction_id") or t.get("id"),
            predicted_class=t["classification"]["predicted_class"],
            confidence=t["classification"]["confidence"],
            alternative_class=t["classification"]["alternative_class"],
            reason_code=t["classification"]["reason_code"],
            direction=t["classification"]["direction"],
            transaction_type=t["classification"]["transaction_type"],
            is_guess=t["classification"]["is_guess"],
        )
        for t in result["transactions"]
    ]

    report = evaluate(predictions, true_labels)
    print_report(report)


if __name__ == "__main__":
    main()
