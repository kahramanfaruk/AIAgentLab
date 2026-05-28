# COST WARNING
# SageMaker real-time endpoints bill continuously per instance hour, which can
# exhaust the 200 USD credit budget quickly. This module therefore provisions
# only an execution role, which has no standing cost, and intentionally does NOT
# define a model or endpoint. Embeddings are served by Bedrock Titan instead
# (pay-per-token, zero idle cost). Add an endpoint here only with a clear budget
# plan; consider SageMaker Serverless Inference, which is pay-per-use, first.

resource "aws_iam_role" "sagemaker_execution" {
  count = var.enable_sagemaker ? 1 : 0
  name  = "${var.project_name}-sagemaker-execution"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect    = "Allow"
        Principal = { Service = "sagemaker.amazonaws.com" }
        Action    = "sts:AssumeRole"
      }
    ]
  })

  tags = var.tags
}
