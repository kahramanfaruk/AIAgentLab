# Deployment

The system deploys as containers; Kubernetes is not used. The serverless
ingestion path runs on AWS Lambda.

## Docker Compose

A single image serves both the API and the UI; the service command selects the
process.

```bash
docker compose up --build
```

This starts the API on port 8000 and the UI on port 8501 using the backends
configured in `.env`. The optional AWS profile adds LocalStack and an
open-source OpenSearch node for exercising the AWS paths at no cost:

```bash
docker compose --profile aws up
```

## AWS

Infrastructure is provisioned with Terraform in `infra/terraform` (S3, DynamoDB,
and the ingestion Lambda by default; OpenSearch and SageMaker are cost-gated and
off by default). See `infra/README.md` for packaging the Lambda and applying the
configuration, and `aws_setup.md` for backend selection and budget guardrails.

## Configuration

All deployment behavior is driven by the environment variables in
`.env.example`. The defaults keep every backend local; switch individual
backends to AWS as needed without code changes.
