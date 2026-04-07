from __future__ import annotations

from dataclasses import dataclass

from agent.generation.llm_client import GroqClient
from agent.ingestion.embedder import LocalEmbedder
from agent.retrieval.retriever import Retriever
from agent.retrieval.vector_store import ChromaVectorStore


@dataclass(slots=True)
class RAGAnswer:
    question: str
    answer: str
    context_chunks: list[dict]


class RAGChain:
    def __init__(
        self,
        retriever: Retriever,
        llm_client: GroqClient,
    ) -> None:
        self.retriever = retriever
        self.llm_client = llm_client

    def ask(self, question: str, top_k: int = 5) -> RAGAnswer:
        results = self.retriever.retrieve(query=question, top_k=top_k)
        context_blocks = []

        for idx, result in enumerate(results, start=1):
            page = result.metadata.get("page_number", "unknown")
            source = result.metadata.get("file_name", "unknown")
            context_blocks.append(
                {
                    "chunk_id": result.chunk_id,
                    "page_number": page,
                    "source": source,
                    "content": result.content,
                    "distance": result.distance,
                }
            )

        prompt = self._build_prompt(question=question, context_blocks=context_blocks)
        answer = self.llm_client.generate(prompt)

        return RAGAnswer(
            question=question,
            answer=answer,
            context_chunks=context_blocks,
        )

    def _build_prompt(self, question: str, context_blocks: list[dict]) -> str:
        if not context_blocks:
            return (
                "You are a helpful AI assistant.\n"
                "No relevant context was found.\n"
                f"Question: {question}\n"
                "Answer: I don't have enough information in the provided documents to answer this confidently."
            )

        formatted_context = "\n\n".join(
            [
                (
                    f"[{block['chunk_id']}] "
                    f"(source: {block['source']}, page: {block['page_number']})\n"
                    f"{block['content']}"
                )
                for block in context_blocks
            ]
        )

        return f"""
You are a careful retrieval-augmented assistant.

Use ONLY the context below.
First identify the most relevant evidence, then answer directly.

If the answer is not clearly supported by the context, say exactly:
"I don't have enough information in the provided documents to answer this confidently."

Rules:
1. Do not use outside knowledge.
2. Keep the answer to 2-4 sentences.
3. Cite chunk IDs inline like [chunk-id].
4. For questions about a paper's purpose, objective, goal, or main contribution, prioritize abstract and introduction evidence.
5. Prefer explicit statements such as "In this work..." and "Our central finding..."
6. Do not mention "most relevant evidence" in the final answer. Just answer the question directly.

Question:
{question}

Context:
{formatted_context}

Answer:
""".strip()

if __name__ == "__main__":
    store = ChromaVectorStore()
    embedder = LocalEmbedder(device="cpu")
    retriever = Retriever(vector_store=store, embedder=embedder)
    llm_client = GroqClient()

    rag = RAGChain(retriever=retriever, llm_client=llm_client)

    question = "What do the authors claim is the main benefit of applying a sigmoid gate after SDPA?"
    result = rag.ask(question=question, top_k=4)

    print("QUESTION:")
    print(result.question)
    print("\nANSWER:")
    print(result.answer)
    print("\nCONTEXT CHUNKS:")
    for chunk in result.context_chunks:
        print(f"- {chunk['chunk_id']} (page {chunk['page_number']})")