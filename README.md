# AIAgentLab

AIAgentLab is an agentic retrieval-augmented generation system for question
answering over documents, with an insurance-operations focus. It combines a
tool-using autonomous agent, pluggable AWS or local backends, a FastAPI service,
and a Streamlit interface in one repository.

The design principle is ports-and-adapters: every external dependency is an
interface with a free local-default adapter and an AWS adapter selected by
configuration. The system runs and is fully testable on a free local stack with
no AWS account, and the same code runs on AWS by changing environment variables.

## Features

- Autonomous agent that plans, uses tools, scores its confidence, and escalates
  to a human when it should not answer on its own
- Rule-based guardrails (PII redaction, blocked topics, output limits), with
  optional Amazon Bedrock Guardrails server-side
- Pluggable backends behind protocols:
  - LLM: Groq (default) or Amazon Bedrock (Claude)
  - Embeddings: Sentence Transformers (default) or Bedrock Titan
  - Vector store: Chroma (default) or OpenSearch
  - Document store: local filesystem (default) or S3
  - Memory and escalations: in-process (default) or DynamoDB
- Serverless ingestion on AWS Lambda, triggered by S3 uploads
- Evaluation framework with retrieval, faithfulness, tool-selection, and
  escalation-decision metrics
- Terraform IaC with cost-gated OpenSearch and SageMaker modules
- FastAPI service, Streamlit UI, Docker Compose, tests, and CI

## Architecture

```text
ingestion -> retrieval -> generation
                 ^             |
                 |          agent (tools, guardrails, escalation)
              memory <---------+
```

- `agent/ingestion`, `agent/retrieval`, `agent/generation`: the RAG pipeline
- `agent/orchestration`: the agent, tools, and guardrails
- `agent/evaluation`: metrics and the evaluation harness
- `agent/storage`, `agent/memory`: document storage and conversation/escalation
  state
- `agent/serverless`: the AWS Lambda ingestion handler
- `api/`, `ui/`, `config/`, `infra/`, `tests/`, `docs/`

See `docs/architecture/overview.md`, `docs/architecture/aws_architecture.md`,
and `docs/guides/agent_overview.md` for details.

## Installation

```bash
git clone https://github.com/kahramanfaruk/AIAgentLab.git
cd AIAgentLab
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Add your `GROQ_API_KEY` to `.env`. The defaults select the free local stack, so
no AWS account is required. To install only the AWS extras, use
`pip install -e ".[aws]"`.

## Configuration

Configuration is centralized in `config/settings.py` and loaded from `.env`.
Backends are selected by switches (all default to the free local stack):

```env
LLM_PROVIDER=groq            # groq | bedrock
EMBEDDING_PROVIDER=local     # local | bedrock
VECTOR_BACKEND=chroma        # chroma | opensearch
STORAGE_BACKEND=local        # local | s3
MEMORY_BACKEND=memory        # memory | dynamodb
```

See `.env.example` for the full set, including AWS, retrieval, and agent
settings.

## Usage

### Streamlit UI

```bash
make run
```

Upload a document, ingest it, and choose a mode: RAG (a single grounded answer)
or Agent (planning, tool use, escalation, and a visible trace).

### FastAPI service

```bash
make api
```

Endpoints: `GET /health`, `POST /ingest`, `POST /ask`, `POST /agent`,
`GET /documents`, `GET /escalations`. Interactive docs are at `/docs`.

### Evaluation

```bash
make eval
```

Runs the agent over `data/eval/insurance_qa.jsonl` and prints aggregate metrics.

## AWS

The live AWS path uses only pay-per-use, near-zero-idle services (S3, DynamoDB,
Lambda, Bedrock). Standing-cost services (managed OpenSearch, SageMaker
endpoints) are demonstrated through the open-source OpenSearch Docker image and
cost-gated Terraform modules rather than run continuously, to fit a small credit
budget. See `docs/architecture/aws_architecture.md` and `docs/guides/aws_setup.md`,
and `infra/README.md` for the Terraform module.

## Docker

```bash
docker compose up --build
```

Starts the API (port 8000) and UI (port 8501). The optional `aws` profile adds
LocalStack and open-source OpenSearch:

```bash
docker compose --profile aws up
```

## Development

```bash
make install   # editable install with dev extras
make lint       # ruff + mypy
make test       # pytest with coverage
make run        # Streamlit UI
make api        # FastAPI service
make eval       # evaluation harness
```

## Project structure

```text
AIAgentLab/
├── agent/
│   ├── evaluation/
│   ├── generation/
│   ├── ingestion/
│   ├── memory/
│   ├── orchestration/
│   ├── retrieval/
│   ├── serverless/
│   └── storage/
├── api/
├── config/
├── data/
├── docs/
├── infra/
│   └── terraform/
├── notebooks/
├── tests/
├── ui/
├── docker-compose.yml
├── Dockerfile
├── Makefile
├── pyproject.toml
└── README.md
```

## Testing

```bash
pytest tests/ -v
```

AWS adapter tests use moto, so they run in-process with no real cloud calls.

## License

This project is licensed under the MIT License. See `LICENSE` for details.
