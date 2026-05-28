"""Prompt templates for the RAG chain and the agent.

Centralizing prompts keeps wording consistent across the API, UI, and agent,
and keeps prompt engineering separate from orchestration logic.
"""

from __future__ import annotations

from typing import Any

INSUFFICIENT_CONTEXT_ANSWER = (
    "I don't have enough information in the provided documents to answer this "
    "confidently."
)

AGENT_SYSTEM_PROMPT = (
    "You are an autonomous assistant supporting insurance operations. You plan "
    "step by step, use tools to gather grounded evidence, and prepare clear, "
    "well-supported answers. You never invent policy facts. When a request "
    "needs human judgment, when required information is missing, or when acting "
    "could carry material risk, you escalate to a human instead of guessing."
)


def build_agent_prompt(
    question: str, tools_description: str, scratchpad: str
) -> str:
    """Build the planner prompt for one agent step.

    The planner must reply with exactly one JSON object: either a tool call or
    a final answer. Keeping the contract strict makes parsing reliable across
    LLM backends.

    Parameters
    ----------
    question : str
        The user question.
    tools_description : str
        Rendered tool catalogue from the tool registry.
    scratchpad : str
        The accumulated thoughts, actions, and observations so far. Empty on the
        first step.

    Returns
    -------
    str
        The rendered planner prompt.
    """
    history = scratchpad.strip() or "(no actions yet)"
    return f"""
You decide the next single step toward answering the question.

Respond with EXACTLY ONE JSON object and nothing else.

To call a tool:
{{"thought": "<your reasoning>", "action": {{"tool": "<tool name>", "input": {{...}}}}}}

To give the final answer (confidence is a number from 0 to 1):
{{"thought": "<your reasoning>", "final_answer": "<answer>", "confidence": <0..1>}}

Guidance:
- Retrieve grounding evidence before answering questions about document content.
- Base the final answer only on tool observations; do not use outside knowledge.
- Set confidence honestly. If evidence is weak or the request needs human
  judgment, set a low confidence or call escalate_to_human.

Available tools:
{tools_description}

Question:
{question}

Scratchpad:
{history}
""".strip()



def build_rag_prompt(question: str, context_blocks: list[dict[str, Any]]) -> str:
    """Build the grounded question-answering prompt.

    Parameters
    ----------
    question : str
        The user question.
    context_blocks : list of dict
        Retrieved context. Each block must contain ``chunk_id``, ``source``,
        ``page_number``, and ``content``.

    Returns
    -------
    str
        The fully rendered prompt. When ``context_blocks`` is empty the prompt
        instructs the model to decline rather than hallucinate.
    """
    if not context_blocks:
        return (
            "You are a helpful AI assistant.\n"
            "No relevant context was found.\n"
            f"Question: {question}\n"
            f"Answer: {INSUFFICIENT_CONTEXT_ANSWER}"
        )

    formatted_context = "\n\n".join(
        f"[{block['chunk_id']}] "
        f"(source: {block['source']}, page: {block['page_number']})\n"
        f"{block['content']}"
        for block in context_blocks
    )

    return f"""
You are a careful retrieval-augmented assistant.

Answer using ONLY the provided context.

Important:
- A document's purpose or objective may be implied by statements such as:
  "In this work..."
  "We conduct..."
  "We investigate..."
  "Our central finding..."
- If the context strongly implies the answer, answer directly in plain language.
- Do not require the exact words "purpose" or "objective" to appear.
- Only decline if the context truly does not support a reasonable answer.

Rules:
1. Do not use outside knowledge.
2. Keep the answer to 2-4 sentences.
3. Cite chunk IDs inline like [chunk-id].
4. For questions about a document's goal or purpose, prioritize early-section evidence.
5. Do not say "the most relevant evidence is..." in the answer.
6. Answer in natural language, not as a retrieval report.

Question:
{question}

Context:
{formatted_context}

Answer:
""".strip()
