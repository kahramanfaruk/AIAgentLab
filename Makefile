.PHONY: install lint test run api eval localstack-up localstack-down tf-validate tf-plan clean

install:
	pip install -e ".[dev]"

lint:
	ruff check . && mypy agent/ api/ config/

test:
	pytest tests/ -v --cov=agent --cov=api --cov-report=term-missing

run:
	streamlit run ui/app.py

api:
	uvicorn api.main:app --reload

eval:
	python -m agent.evaluation data/eval/insurance_qa.jsonl

localstack-up:
	docker compose --profile aws up -d localstack opensearch

localstack-down:
	docker compose --profile aws down

tf-validate:
	cd infra/terraform && terraform init -backend=false && terraform validate

tf-plan:
	cd infra/terraform && terraform plan

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .pytest_cache .coverage htmlcov dist build
