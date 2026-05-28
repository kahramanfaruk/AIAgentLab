# Agent overview

The agent turns the RAG system into an autonomous, tool-using assistant suited
to insurance operations. It plans, gathers evidence, decides whether it can
answer reliably, and escalates to a human when it cannot.

## Loop

The agent (`agent/orchestration/agent.py`) plans one step at a time using
structured JSON prompting, which keeps the loop identical across LLM backends
and trivially testable. Each step the planner returns either a tool call or a
final answer:

1. Apply input guardrails; escalate immediately if the input is blocked.
2. Build a planner prompt from the question, the tool catalogue, and the running
   scratchpad of observations.
3. Ask the LLM for the next step.
4. If it is a tool call, run the tool and append the observation.
5. If it is a final answer, validate it and decide answer-or-escalate.
6. Stop after `agent_max_steps`; escalate if no answer was produced.

## Tools

Tools live in `agent/orchestration/tools.py` and are exposed through a registry:

- `retrieve_documents`: retrieves grounding context from the vector store.
- `lookup_policy_clause`: deterministic lookup of structured policy clauses.
- `escalate_to_human`: requests a human handoff with a reason.

The registry renders tool names, descriptions, and input schemas into the
planner prompt, so adding a tool makes it available to the agent automatically.

## Guardrails

`agent/orchestration/guardrails.py` is the always-on safety layer. It redacts
common PII (email, phone, IBAN, card numbers) and blocks configured topics and
oversized outputs. When the Bedrock backend is configured with a Guardrail
identifier, that managed guardrail also runs server-side; the two are
complementary.

## Human-in-the-loop escalation

The agent escalates when any of these hold: an input or output guardrail blocks
the text, the final confidence is below `escalation_confidence_threshold`, a
tool requests a handoff, or the step budget is exhausted. Escalations are
persisted through the memory port (in-process locally, DynamoDB on AWS) and
surfaced via `GET /escalations`. The confidence threshold is the core
answer-or-escalate decision gate.

## Evaluation

`agent/evaluation` runs the agent over a labeled dataset
(`data/eval/insurance_qa.jsonl`) and reports retrieval hit-rate, citation
validity (a faithfulness proxy), tool-selection accuracy, keyword coverage, and
escalation-decision accuracy. Run it with `make eval`.
