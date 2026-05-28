# Infrastructure (Terraform)

This module provisions the AWS backends for AIAgentLab. It is designed to stay
within a small credit budget (200 USD), so it separates pay-per-use services
from standing-cost services.

## What is provisioned by default

| Resource | Service | Cost profile |
|----------|---------|--------------|
| Document bucket | S3 | Pay-per-use, negligible idle |
| State table (`pk`/`sk`, on-demand) | DynamoDB | Pay-per-request, negligible idle |
| Ingestion function + S3 trigger | Lambda | Pay-per-invocation, no idle |
| Least-privilege execution role | IAM | No cost |

Generation and embeddings use Amazon Bedrock (pay-per-token, zero idle cost),
which needs no provisioned infrastructure.

## What is gated off by default (standing cost)

- `enable_opensearch` (default `false`): a managed OpenSearch domain bills per
  hour while it exists. During development the open-source OpenSearch Docker
  image in `docker-compose.yml` provides the same k-NN API for free.
- `enable_sagemaker` (default `false`): provisions only an execution role (no
  standing cost). Endpoints are intentionally not defined; embeddings are served
  by Bedrock Titan instead.

Enable them only deliberately, for example
`terraform apply -var="enable_opensearch=true"`.

## Build the Lambda package

The Lambda handler is `agent.serverless.ingestion_handler.handler`. Package the
`agent/` and `config/` modules together with their dependencies into
`build/lambda.zip` (a Lambda layer or container image is recommended in
production because the dependency set is large). Point `lambda_zip_path` at the
archive.

## Local run against LocalStack

```bash
export TF_VAR_aws_endpoint_url=http://localhost:4566
terraform init
terraform validate
terraform plan
terraform apply
```

With `aws_endpoint_url` set, the provider targets LocalStack and skips real
credential and account checks, so this costs nothing.

## Real AWS deploy

Leave `aws_endpoint_url` empty, configure AWS credentials, then
`terraform init && terraform plan && terraform apply`. Review the plan to
confirm no standing-cost resources are created unless you enabled them.
