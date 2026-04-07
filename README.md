# AIAgentLab

AIAgentLab is a modular retrieval-augmented generation system for question answering over uploaded documents. It combines local embeddings, ChromaDB retrieval, Groq-based answer generation, a Streamlit user interface, and a FastAPI service layer in a single repository.

The project is designed as a practical, extensible foundation for research-paper question answering and document-grounded AI assistants. Users can upload a document, index it locally, ask natural-language questions, and inspect the retrieved source chunks used to generate each answer.

## Features

- Document ingestion for PDF and text-based sources
- Local embedding generation with Sentence Transformers
- ChromaDB vector storage for semantic retrieval
- Groq-based grounded answer generation
- Streamlit interface for interactive document Q&A
- FastAPI backend layer for service-oriented integration
- Modular project structure for experimentation and extension
- Test, Docker, and CI-ready repository scaffold

## Architecture

The repository is organized into clear layers:

- `agent/ingestion` for loading, parsing, chunking, and embedding documents
- `agent/retrieval` for vector storage, retrieval, and ranking logic
- `agent/generation` for prompt construction and LLM interaction
- `agent/memory` for chat-history-related utilities
- `api/` for FastAPI endpoints and schemas
- `ui/` for the Streamlit application
- `config/` for shared settings and logging
- `tests/` for unit and integration tests
- `docs/` for architecture and usage documentation

## Current status

The current version supports:
- local document ingestion,
- local semantic retrieval,
- grounded answer generation with Groq,
- and a working Streamlit interface for question answering over uploaded PDF documents.

Gemini support is not included in the current v0.1 working path. If added later, it should use the modern `google-genai` SDK rather than the deprecated `google-generativeai` package [web:118][web:225].

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/kahramanfaruk/AIAgentLab.git
cd AIAgentLab
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Create the environment file

```bash
cp .env.example .env
```

Add your Groq API key to `.env`.

### 4. Install dependencies

```bash
pip install -e ".[dev]"
```

## Configuration

The project currently uses the following core environment variables:

```env
GROQ_API_KEY=your_groq_api_key_here
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
VECTOR_DB_PATH=./data/vector_db
CHROMA_COLLECTION_NAME=rag_documents
CHUNK_SIZE=512
CHUNK_OVERLAP=64
TOP_K_RESULTS=6
HF_TOKEN=
```

## Usage

### Run the Streamlit application

```bash
streamlit run ui/app.py
```

Then open the local Streamlit URL in your browser, upload a PDF, ingest it, and ask questions about the document.

### Example questions

- What is the main objective of this paper?
- What method do the authors propose?
- What are the key findings?
- What limitation does the paper discuss?
- What evidence supports the main claim?

## FastAPI backend

The repository also includes a FastAPI layer intended to expose ingestion and question-answering endpoints.

Planned v0.1 backend endpoints:
- `GET /health`
- `POST /ingest`
- `POST /ask`
- `GET /documents`

Once implemented, the API can be started with:

```bash
uvicorn api.main:app --reload
```

## Project structure

```text
AIAgentLab/
├── agent/
│   ├── generation/
│   ├── ingestion/
│   ├── memory/
│   └── retrieval/
├── api/
├── config/
├── data/
│   ├── processed/
│   ├── raw/
│   └── vector_db/
├── docs/
├── notebooks/
├── tests/
├── ui/
├── .env.example
├── .gitignore
├── Makefile
├── pyproject.toml
└── README.md
```

## Development commands

```bash
make install
make lint
make test
make run
```

## Docker

Docker and Docker Compose are included in the repository scaffold and will be used to run the API and UI as separate services. This is part of the planned one-week completion roadmap.

The intended future command is:

```bash
docker compose up --build
```

## Testing

Run the test suite with:

```bash
pytest tests/ -v
```

Or use:

```bash
make test
```

## Roadmap

### v0.1
- Stable Streamlit document Q&A flow
- Clean configuration management
- FastAPI service endpoints
- Unit and integration tests
- Dockerized local development
- CI for linting and testing
- Professional project documentation

### v0.2
- Optional Gemini provider using `google-genai`
- Improved reranking
- Multi-document collections
- Better evaluation tooling
- Conversation memory improvements
- Deployment guide

## Limitations

- Retrieval quality depends on document parsing and chunking quality.
- The current system is optimized for research-paper-style PDFs.
- Long or noisy PDFs may require more robust preprocessing and reranking.
- FastAPI and Docker are scaffolded but still being completed in the current development phase.

## Contributing

Contributions, issue reports, and suggestions are welcome. See `CONTRIBUTING.md` for contribution guidance.

## License

This project is licensed under the MIT License. See `LICENSE` for details.