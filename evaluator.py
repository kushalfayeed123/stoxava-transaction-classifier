"""
Accuracy reporting.

Reports overall accuracy, per-class precision/recall, and a confusion
matrix. Overall accuracy is the headline "80%+" metric; per-class numbers
matter more in practice because overall accuracy can hide a class the model
is bad at if that class is rare in the sample -- flag this to stakeholders
when presenting results.
"""

from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass

from classifier import Prediction


@dataclass
class EvalReport:
    overall_accuracy: float
    n_total: int
    n_correct: int
    n_needs_review: int
    per_class: dict[str, dict[str, float]]  # class -> {precision, recall, support}
    confusion: dict[str, dict[str, int]]  # true_class -> {predicted_class: count}


def evaluate(predictions: list[Prediction], true_labels: dict[str, str]) -> EvalReport:
    n_total = len(predictions)
    n_correct = 0
    n_needs_review = 0

    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    support = defaultdict(int)

    for p in predictions:
        true_cls = true_labels.get(p.transaction_id, "Uncategorized")
        support[true_cls] += 1
        confusion[true_cls][p.predicted_class] += 1

        if p.needs_review:
            n_needs_review += 1

        if p.predicted_class == true_cls:
            n_correct += 1
            tp[true_cls] += 1
        else:
            fp[p.predicted_class] += 1
            fn[true_cls] += 1

    per_class: dict[str, dict[str, float]] = {}
    for cls in support:
        precision_denom = tp[cls] + fp[cls]
        recall_denom = tp[cls] + fn[cls]
        per_class[cls] = {
            "precision": (tp[cls] / precision_denom) if precision_denom else 0.0,
            "recall": (tp[cls] / recall_denom) if recall_denom else 0.0,
            "support": support[cls],
        }

    return EvalReport(
        overall_accuracy=(n_correct / n_total) if n_total else 0.0,
        n_total=n_total,
        n_correct=n_correct,
        n_needs_review=n_needs_review,
        per_class=per_class,
        confusion={k: dict(v) for k, v in confusion.items()},
    )


def print_report(report: EvalReport) -> None:
    print(f"\nOverall accuracy: {report.overall_accuracy:.1%} "
          f"({report.n_correct}/{report.n_total} correct)")
    print(f"Flagged for human review (confidence < 0.6): {report.n_needs_review}")
    print("\nPer-class precision / recall / support:")
    for cls, m in sorted(report.per_class.items()):
        print(f"  {cls:<16} precision={m['precision']:.2f}  recall={m['recall']:.2f}  support={int(m['support'])}")