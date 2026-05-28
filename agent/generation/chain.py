"""The retrieval-augmented generation chain.

The chain retrieves grounding context, renders the RAG prompt, and asks the
configured LLM backend for a grounded answer. It depends only on the
:class:`~agent.generation.base.LLMClient` protocol and the retriever, so it is
backend-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent.generation.base import LLMClient
from agent.generation.prompts import build_rag_prompt
from agent.retrieval.retriever import Retriever


@dataclass(slots=True)
class RAGAnswer:
    """A grounded answer and the context used to produce it.

    Attributes
    ----------
    question : str
        The original question.
    answer : str
        The generated answer.
    context_chunks : list of dict
        The retrieved context blocks used as grounding.
    """

    question: str
    answer: str
    context_chunks: list[dict[str, Any]]


class RAGChain:
    """Coordinate retrieval and generation for grounded question answering.

    Parameters
    ----------
    retriever : Retriever
        Component that returns ranked context for a question.
    llm_client : LLMClient
        Backend used to generate the answer.
    """

    def __init__(self, retriever: Retriever, llm_client: LLMClient) -> None:
        self.retriever = retriever
        self.llm_client = llm_client

    def ask(self, question: str, top_k: int = 5) -> RAGAnswer:
        """Answer ``question`` using retrieved context.

        Parameters
        ----------
        question : str
            The user question.
        top_k : int, optional
            Number of context chunks to retrieve.

        Returns
        -------
        RAGAnswer
            The answer together with the context blocks used.
        """
        results = self.retriever.retrieve(query=question, top_k=top_k)
        context_blocks = [
            {
                "chunk_id": result.chunk_id,
                "page_number": result.metadata.get("page_number", "unknown"),
                "source": result.metadata.get("file_name", "unknown"),
                "content": result.content,
                "distance": result.distance,
            }
            for result in results
        ]

        prompt = build_rag_prompt(question=question, context_blocks=context_blocks)
        answer = self.llm_client.generate(prompt)

        return RAGAnswer(
            question=question,
            answer=answer,
            context_chunks=context_blocks,
        )
