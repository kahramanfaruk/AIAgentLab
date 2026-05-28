"""Command-line entry point for the evaluation harness.

Usage
-----
``python -m agent.evaluation [dataset_path]``

Builds an agent from the active settings, runs it over the dataset (defaulting
to ``data/eval/insurance_qa.jsonl``), and prints the aggregate report. This
requires a configured LLM backend, so it is a development command rather than a
CI step.
"""

from __future__ import annotations

import sys
from pathlib import Path

from agent.evaluation.evaluator import evaluate, load_dataset
from agent.orchestration import build_agent
from config import get_settings

_DEFAULT_DATASET = Path("data/eval/insurance_qa.jsonl")


def main(argv: list[str] | None = None) -> int:
    """Run the evaluation harness.

    Parameters
    ----------
    argv : list of str or None, optional
        Command-line arguments; the first is an optional dataset path.

    Returns
    -------
    int
        Process exit code (``0`` on success).
    """
    args = sys.argv[1:] if argv is None else argv
    dataset_path = Path(args[0]) if args else _DEFAULT_DATASET

    settings = get_settings()
    agent = build_agent(settings)
    dataset = load_dataset(dataset_path)
    report = evaluate(agent, dataset)
    print(report.format())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
