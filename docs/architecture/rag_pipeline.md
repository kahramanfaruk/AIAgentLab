# RAG pipeline

The retrieval-augmented generation pipeline grounds answers in indexed document
content.

## Ingestion

1. Load: `agent/ingestion/loader.py` parses PDF, text, and DOCX files into
   per-unit `Document` objects (PDF pages are separate units).
2. Chunk: `agent/ingestion/chunker.py` splits documents into overlapping chunks,
   sized by `CHUNK_SIZE` and `CHUNK_OVERLAP`.
3. Embed: the configured embedder (`LocalEmbedder` or `BedrockEmbedder`) turns
   chunks into vectors.
4. Index: the configured vector store (`ChromaVectorStore` or
   `OpenSearchVectorStore`) stores the vectors.

The shared `index_document_bytes` helper (`agent/ingestion/pipeline.py`) runs
these steps and is reused by the API ingest endpoint and the Lambda handler.

## Query

1. Embed the query with the same embedder.
2. Retrieve: `agent/retrieval/retriever.py` over-fetches candidates from the
   vector store and applies a lightweight heuristic re-ranking tuned for
   document purpose, method, and finding questions.
3. Generate: `agent/generation/chain.py` renders the grounding prompt
   (`agent/generation/prompts.py`) and asks the configured LLM for an answer
   that cites chunk identifiers.

The agent reuses the same retriever as a tool and adds planning, guardrails, and
escalation on top; see `../guides/agent_overview.md`.
