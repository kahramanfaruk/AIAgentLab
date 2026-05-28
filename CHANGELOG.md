# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project aims to follow Semantic Versioning for tagged releases [web:361][web:364].

## [Unreleased]

### Added
- Initial modular repository structure for ingestion, retrieval, generation, memory, API, UI, tests, and docs.
- Local document ingestion pipeline for PDF and text sources.
- ChromaDB-based retrieval pipeline.
- Groq-based generation client.
- Streamlit-based document question answering interface.
- Centralized configuration with `pydantic-settings` and per-backend switches.
- Ports-and-adapters backends with free local defaults and AWS adapters: Bedrock LLM, Bedrock Titan embeddings, OpenSearch vector store, S3 document store, and DynamoDB memory and escalation store.
- Autonomous agent loop with a tool registry, rule-based guardrails, model-confidence scoring, and human-in-the-loop escalation.
- Evaluation framework with retrieval, faithfulness, tool-selection, and escalation-decision metrics plus a synthetic insurance dataset.
- Serverless ingestion Lambda handler and Terraform IaC for S3, DynamoDB, and Lambda, with cost-gated OpenSearch and SageMaker modules.
- FastAPI endpoints for health, ingest, ask, agent, documents, and escalations.
- Dockerfile and Docker Compose stack, including an optional AWS profile with LocalStack and open-source OpenSearch.
- Test suite covering adapters (via moto), guardrails, the agent loop, evaluation, the API, and the ingestion flow.

### Changed
- Standardized the active v0.1 LLM path around Groq, now pluggable with Amazon Bedrock.
- The RAG chain and retriever depend on backend-agnostic protocols rather than concrete Chroma and Groq classes.
- The Streamlit UI adds an agent mode that shows the reasoning trace and escalations.
- Improved retrieval behavior for paper-purpose and objective-style questions.
- Refined prompt instructions for grounded paper summarization.

### Removed
- Deprecated Gemini dependency path based on `google-generativeai`.

## [0.1.0] - 2026-04-07

### Added
- Initial working MVP for document-grounded RAG question answering.
- Streamlit interface for PDF upload, ingestion, and question answering.
- Local embeddings using Sentence Transformers.
- Vector retrieval with ChromaDB.
- Prompted answer generation with source-aware grounding.

### Known limitations
- FastAPI layer is scaffolded and still being completed.
- Docker and CI are scaffolded and still being completed.
- Retrieval quality is currently optimized for research-paper-style PDFs.