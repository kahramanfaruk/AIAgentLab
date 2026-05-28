"""Dataset-level evaluation harness for the agent.

The harness runs the agent over a labeled dataset, scores each run with the
per-item metrics, and aggregates the results into a report. It is the decision
framework's measurement layer: it quantifies answer quality, retrieval quality,
faithfulness, and the correctness of the escalate-or-answer decision.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from agent.evaluation.metrics import (
    citation_validity,
    escalation_correct,
    keyword_coverage,
    retrieval_hit,
    tool_selection_correct,
)
from agent.orchestration.agent import Agent


@dataclass(slots=True)
class EvalItem:
    """A single labeled evaluation example.

    Attributes
    ----------
    question : str
        The question to ask the agent.
    expected_keywords : list of str
        Keywords expected in a correct answer.
    expected_escalation : bool
        Whether the agent should escalate this item.
    relevant_sources : list of str
        Source file names considered relevant for retrieval.
    expected_tool : str or None
        The tool expected to be used, if any.
    """

    question: str
    expected_keywords: list[str] = field(default_factory=list)
    expected_escalation: bool = False
    relevant_sources: list[str] = field(default_factory=list)
    expected_tool: str | None = None


@dataclass(slots=True)
class ItemResult:
    """The scored outcome of one evaluation item.

    Attributes
    ----------
    question : str
        The evaluated question.
    answer : str
        The agent's answer.
    escalated : bool
        Whether the run escalated.
    metrics : dict
        Per-item metric scores.
    """

    question: str
    answer: str
    escalated: bool
    metrics: dict[str, float]


@dataclass(slots=True)
class EvaluationReport:
    """Aggregate evaluation results.

    Attributes
    ----------
    items : list of ItemResult
        Per-item scored results.
    aggregates : dict
        Mean of each metric across applicable items.
    """

    items: list[ItemResult]
    aggregates: dict[str, float]

    def format(self) -> str:
        """Render the report as a readable text summary.

        Returns
        -------
        str
            A multi-line summary of aggregate metrics and per-item escalation
            decisions.
        """
        lines = ["Evaluation report", "=" * 40, f"items: {len(self.items)}", ""]
        lines.append("Aggregate metrics:")
        for name, value in sorted(self.aggregates.items()):
            lines.append(f"  {name}: {value:.3f}")
        lines.append("")
        lines.append("Per-item decisions:")
        for item in self.items:
            lines.append(f"  - escalated={item.escalated!s:<5} | {item.question}")
        return "\n".join(lines)


def load_dataset(path: str | Path) -> list[EvalItem]:
    """Load a JSON Lines evaluation dataset.

    Parameters
    ----------
    path : str or Path
        Path to a ``.jsonl`` file with one JSON object per line.

    Returns
    -------
    list of EvalItem
        Parsed evaluation items.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If a line is not valid JSON or is missing the ``question`` field.
    """
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Evaluation dataset not found: {dataset_path}")

    items: list[EvalItem] = []
    for line_number, raw_line in enumerate(
        dataset_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON on line {line_number} of {dataset_path}: {exc}"
            ) from exc
        if "question" not in record:
            raise ValueError(
                f"Missing 'question' on line {line_number} of {dataset_path}."
            )
        items.append(
            EvalItem(
                question=record["question"],
                expected_keywords=record.get("expected_keywords", []),
                expected_escalation=record.get("expected_escalation", False),
                relevant_sources=record.get("relevant_sources", []),
                expected_tool=record.get("expected_tool"),
            )
        )
    return items


def evaluate(agent: Agent, dataset: list[EvalItem]) -> EvaluationReport:
    """Run the agent over a dataset and score every item.

    Parameters
    ----------
    agent : Agent
        The agent under evaluation.
    dataset : list of EvalItem
        The labeled examples.

    Returns
    -------
    EvaluationReport
        Per-item results and aggregate metrics.
    """
    item_results: list[ItemResult] = []
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}

    def _record(name: str, value: float) -> None:
        sums[name] = sums.get(name, 0.0) + value
        counts[name] = counts.get(name, 0) + 1

    for item in dataset:
        result = agent.run(item.question)

        metrics: dict[str, float] = {
            "escalation_correct": escalation_correct(
                result.escalated, item.expected_escalation
            ),
            "tool_selection_correct": tool_selection_correct(
                result.trace, item.expected_tool
            ),
        }

        if item.relevant_sources:
            metrics["retrieval_hit"] = retrieval_hit(
                result.context_chunks, item.relevant_sources
            )
        # Answer-quality metrics apply only when an answer is expected.
        if not item.expected_escalation:
            metrics["keyword_coverage"] = keyword_coverage(
                result.answer, item.expected_keywords
            )
            metrics["citation_validity"] = citation_validity(
                result.answer, result.context_chunks
            )

        for name, value in metrics.items():
            _record(name, value)

        item_results.append(
            ItemResult(
                question=item.question,
                answer=result.answer,
                escalated=result.escalated,
                metrics=metrics,
            )
        )

    aggregates = {name: sums[name] / counts[name] for name in sums}
    return EvaluationReport(items=item_results, aggregates=aggregates)
