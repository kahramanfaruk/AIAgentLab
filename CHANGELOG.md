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

### Changed
- Standardized the active v0.1 LLM path around Groq.
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