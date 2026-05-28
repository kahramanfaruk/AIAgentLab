"""Evaluation package: metrics and the dataset evaluation harness."""

from __future__ import annotations

from agent.evaluation.evaluator import (
    EvalItem,
    EvaluationReport,
    ItemResult,
    evaluate,
    load_dataset,
)
from agent.evaluation.metrics import (
    citation_validity,
    escalation_correct,
    keyword_coverage,
    retrieval_hit,
    tool_selection_correct,
)

__all__ = [
    "EvalItem",
    "EvaluationReport",
    "ItemResult",
    "evaluate",
    "load_dataset",
    "citation_validity",
    "escalation_correct",
    "keyword_coverage",
    "retrieval_hit",
    "tool_selection_correct",
]
