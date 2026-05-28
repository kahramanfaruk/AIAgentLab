# Architecture overview

AIAgentLab is an agentic retrieval-augmented generation system with pluggable
backends. It runs on a free local stack by default and on AWS through
configuration, with no code changes.

## Layers

- `agent/ingestion`: document loading, chunking, embedding, and the shared
  indexing pipeline. Embedding has a local default and a Bedrock adapter.
- `agent/retrieval`: vector storage and retrieval. Chroma is the default;
  OpenSearch is the AWS adapter. The retriever depends on a `VectorStore`
  protocol, not a concrete store.
- `agent/generation`: prompts, the RAG chain, and the LLM clients (Groq default,
  Bedrock adapter) behind an `LLMClient` protocol.
- `agent/storage`: document storage (local filesystem default, S3 adapter).
- `agent/memory`: chat history and the escalation store (in-process default,
  DynamoDB adapter).
- `agent/orchestration`: the autonomous agent, its tools, and guardrails.
- `agent/evaluation`: metrics and the dataset evaluation harness.
- `agent/serverless`: the AWS Lambda ingestion handler.
- `api`: the FastAPI service and its wiring.
- `ui`: the Streamlit application.
- `config`: the single source of configuration truth.
- `infra`: Terraform IaC.

## Design principle

Every external dependency is a port (a `typing.Protocol`) with a free local
adapter and an AWS adapter. Factories in each package (`build_llm_client`,
`build_embedder`, `build_vector_store`, `build_document_store`, `build_memory`,
`build_agent`) select the adapter from `Settings`. This keeps the system
runnable and testable offline while making the AWS path real and
Terraform-provisioned.

See `aws_architecture.md` for the AWS topology and cost rationale, and
`../guides/agent_overview.md` for the agent loop.
