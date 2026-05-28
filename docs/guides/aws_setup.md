# AWS setup

This guide covers running the AWS-pluggable backends, both for free against
LocalStack and against a real AWS account, while staying inside a small credit
budget. See `docs/architecture/aws_architecture.md` for the cost rationale.

## Selecting backends

Backends are chosen entirely through environment variables (see `.env.example`).
The defaults select the free local stack, so no AWS account is needed. To move a
backend to AWS, change its switch:

```env
LLM_PROVIDER=bedrock
EMBEDDING_PROVIDER=bedrock
EMBEDDING_DIM=1024            # Titan Text Embeddings v2
VECTOR_BACKEND=opensearch
STORAGE_BACKEND=s3
MEMORY_BACKEND=dynamodb
```

## Free local AWS path with LocalStack

LocalStack emulates S3, DynamoDB, and Lambda locally at no cost. Start it and an
open-source OpenSearch node with the optional Docker profile:

```bash
docker compose --profile aws up -d localstack opensearch
```

Point the AWS clients at LocalStack and OpenSearch at the local container:

```env
AWS_ENDPOINT_URL=http://localhost:4566
STORAGE_BACKEND=s3
MEMORY_BACKEND=dynamodb
VECTOR_BACKEND=opensearch
OPENSEARCH_HOST=localhost
```

Provision the resources with Terraform (the same configuration targets
LocalStack when `aws_endpoint_url` is set):

```bash
cd infra/terraform
terraform init
terraform validate
TF_VAR_aws_endpoint_url=http://localhost:4566 terraform apply
```

Bedrock is not emulated by LocalStack, so keep `LLM_PROVIDER=groq` and
`EMBEDDING_PROVIDER=local` for fully offline runs, or use real Bedrock (it has
zero idle cost) for the generation and embedding steps.

## Real AWS path

1. Configure AWS credentials and leave `AWS_ENDPOINT_URL` empty.
2. Request access to the Bedrock models you intend to use (Claude and Titan).
3. Build the Lambda package (see `infra/README.md`) and run
   `terraform init && terraform plan && terraform apply`.
4. Keep `enable_opensearch` and `enable_sagemaker` off unless you have a budget
   plan; they carry standing cost.

## Budget guardrails

- Prefer Claude Haiku and Titan v2; both are inexpensive per request.
- Do not leave a managed OpenSearch domain or a SageMaker endpoint running.
- Use DynamoDB on-demand billing (the Terraform default) so idle cost is near
  zero.
