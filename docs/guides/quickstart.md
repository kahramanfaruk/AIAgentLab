# Quickstart

This gets the system running on the free local stack (Chroma, Groq, local
filesystem, in-process memory). No AWS account is required.

## 1. Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## 2. Configure

```bash
cp .env.example .env
```

Add a `GROQ_API_KEY` to `.env` for generation. The defaults select the local
backends, so nothing else is required.

## 3. Run the UI

```bash
make run
```

Upload a document, click ingest, pick a mode, and ask a question:

- RAG mode: a single grounded answer with its sources.
- Agent mode: the agent plans, uses tools, may escalate, and shows its trace.

## 4. Run the API

```bash
make api
```

Then call the endpoints, for example:

```bash
curl localhost:8000/health
curl -X POST localhost:8000/agent -H 'content-type: application/json' \
  -d '{"question": "What is the deductible?"}'
```

## 5. Quality checks

```bash
make lint
make test
make eval
```

To exercise the AWS-pluggable backends for free, see `aws_setup.md`.
