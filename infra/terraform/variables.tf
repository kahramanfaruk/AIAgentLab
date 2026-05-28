variable "aws_region" {
  description = "AWS region for all resources."
  type        = string
  default     = "eu-central-1"
}

variable "project_name" {
  description = "Name prefix applied to created resources."
  type        = string
  default     = "aiagentlab"
}

variable "aws_endpoint_url" {
  description = "Override endpoint for all AWS services (set to a LocalStack URL for local runs). Empty targets real AWS."
  type        = string
  default     = ""
}

variable "s3_bucket_name" {
  description = "Name of the document bucket."
  type        = string
  default     = "aiagentlab-documents"
}

variable "dynamodb_table_name" {
  description = "Name of the single-table state store."
  type        = string
  default     = "aiagentlab-state"
}

variable "lambda_function_name" {
  description = "Name of the ingestion Lambda function."
  type        = string
  default     = "aiagentlab-ingestion"
}

variable "lambda_zip_path" {
  description = "Path to the packaged Lambda deployment archive. Build it before apply (see infra/README.md)."
  type        = string
  default     = "build/lambda.zip"
}

variable "lambda_runtime" {
  description = "Lambda Python runtime."
  type        = string
  default     = "python3.11"
}

variable "lambda_handler" {
  description = "Lambda handler entry point."
  type        = string
  default     = "agent.serverless.ingestion_handler.handler"
}

variable "lambda_timeout" {
  description = "Lambda timeout in seconds."
  type        = number
  default     = 120
}

variable "lambda_memory_size" {
  description = "Lambda memory in MB."
  type        = number
  default     = 1024
}

variable "bedrock_model_id" {
  description = "Bedrock model id used for generation."
  type        = string
  default     = "anthropic.claude-3-5-haiku-20241022-v1:0"
}

variable "bedrock_embedding_model_id" {
  description = "Bedrock model id used for embeddings."
  type        = string
  default     = "amazon.titan-embed-text-v2:0"
}

# COST CONTROL: OpenSearch and SageMaker carry standing (hourly) cost and are
# disabled by default to protect the 200 USD credit budget. Enable deliberately.
variable "enable_opensearch" {
  description = "Provision a managed OpenSearch domain (incurs hourly cost while it exists)."
  type        = bool
  default     = false
}

variable "enable_sagemaker" {
  description = "Provision a SageMaker execution role (no standing cost; endpoints are intentionally not defined)."
  type        = bool
  default     = false
}

variable "opensearch_instance_type" {
  description = "OpenSearch data node instance type (used only when enable_opensearch is true)."
  type        = string
  default     = "t3.small.search"
}

variable "opensearch_volume_size" {
  description = "OpenSearch EBS volume size in GB (used only when enable_opensearch is true)."
  type        = number
  default     = 10
}

variable "tags" {
  description = "Tags applied to all resources."
  type        = map(string)
  default = {
    project = "aiagentlab"
    managed = "terraform"
  }
}
