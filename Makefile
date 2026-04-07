.PHONY: install lint test run ingest clean

install:
	pip install -e ".[dev]"

lint:
	ruff check . && mypy agent/ api/ config/

test:
	pytest tests/ -v --cov=agent --cov-report=term-missing

run:
	streamlit run ui/app.py

ingest:
	python -m agent.ingestion.loader --source data/raw/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .pytest_cache .coverage htmlcov dist build
