# AWS architecture and cost rationale

AIAgentLab uses a ports-and-adapters design. Every external dependency is an
interface with a free local-default adapter and an AWS adapter selected by
configuration. The application code never changes between local and cloud; only
the backend switches in `config/settings.py` change.

## Backends

| Port | Local default (free) | AWS adapter |
|------|----------------------|-------------|
| LLM | Groq (`GroqClient`) | Amazon Bedrock (`BedrockClient`, Claude) |
| Embeddings | Sentence Transformers (`LocalEmbedder`) | Bedrock Titan (`BedrockEmbedder`) |
| Vector store | Chroma (`ChromaVectorStore`) | OpenSearch (`OpenSearchVectorStore`) |
| Document store | Local filesystem (`LocalDocumentStore`) | S3 (`S3DocumentStore`) |
| Memory / escalations | In-process (`InMemoryConversationStore`) | DynamoDB (`DynamoDBConversationStore`) |
| Ingestion compute | In-process pipeline | AWS Lambda (`agent.serverless.ingestion_handler`) |

## Component flow

```text
            upload                         S3:ObjectCreated
   client ---------> S3 (documents) ---------------------------> Lambda (ingestion)
                                                                     |
                                          load -> chunk -> embed (Bedrock Titan)
                                                                     |
                                                                     v
                                              OpenSearch (k-NN vector index)
                                                                     ^
   client --- POST /ask or /agent ---> FastAPI / Streamlit ---------- retrieve
                                            |                          |
                                            |  generate (Bedrock Claude + Guardrails)
                                            v
                                  DynamoDB (chat history, escalations, doc metadata)
```

## Cost rationale (200 USD credit budget)

The binding constraint is idle (standing) cost, not per-call cost. The live AWS
path uses only pay-per-use services with negligible idle cost:

- S3: pay per storage and request; about 0.023 USD per GB after the free tier.
- DynamoDB on-demand: pay per request; 25 GB of storage is always free.
- Lambda: 1,000,000 free requests and 400,000 GB-seconds per month.
- Bedrock: pay per token, with zero idle cost; Claude Haiku and Titan v2 cost
  fractions of a cent per request at demo volumes.

Standing-cost services are deliberately not run continuously:

- Managed OpenSearch bills per hour while it exists (a small node is roughly
  25 USD per month; OpenSearch Serverless has a far higher minimum near 345 USD
  per month). The open-source OpenSearch Docker image provides the identical
  k-NN API for free during development, and the managed domain is gated behind
  the `enable_opensearch` Terraform variable (default off).
- SageMaker real-time endpoints bill continuously per instance hour. Embeddings
  are served by Bedrock Titan instead; the Terraform module provisions only an
  execution role (no standing cost) behind `enable_sagemaker` (default off).

The result evidences every service named in the target role (Bedrock, S3,
Lambda, DynamoDB, OpenSearch, IaC) while keeping the credit budget effectively
intact, and keeps the whole system runnable and CI-testable with no AWS account.
