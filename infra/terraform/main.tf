# Core, budget-safe infrastructure: S3, DynamoDB, and the ingestion Lambda.
# Every resource here is pay-per-use with negligible idle cost.

resource "aws_s3_bucket" "documents" {
  bucket = var.s3_bucket_name
  tags   = var.tags
}

resource "aws_s3_bucket_public_access_block" "documents" {
  bucket                  = aws_s3_bucket.documents.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_dynamodb_table" "state" {
  name         = var.dynamodb_table_name
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "pk"
  range_key    = "sk"

  attribute {
    name = "pk"
    type = "S"
  }

  attribute {
    name = "sk"
    type = "S"
  }

  tags = var.tags
}

# Least-privilege execution role for the ingestion Lambda.
resource "aws_iam_role" "lambda" {
  name = "${var.project_name}-lambda-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect    = "Allow"
        Principal = { Service = "lambda.amazonaws.com" }
        Action    = "sts:AssumeRole"
      }
    ]
  })

  tags = var.tags
}

locals {
  base_statements = [
    {
      Effect = "Allow"
      Action = [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents",
      ]
      Resource = "arn:aws:logs:*:*:*"
    },
    {
      Effect   = "Allow"
      Action   = ["s3:GetObject"]
      Resource = "${aws_s3_bucket.documents.arn}/*"
    },
    {
      Effect = "Allow"
      Action = [
        "dynamodb:PutItem",
        "dynamodb:GetItem",
        "dynamodb:Query",
      ]
      Resource = aws_dynamodb_table.state.arn
    },
    {
      Effect   = "Allow"
      Action   = ["bedrock:InvokeModel"]
      Resource = "arn:aws:bedrock:*::foundation-model/*"
    },
  ]

  opensearch_statements = var.enable_opensearch ? [
    {
      Effect   = "Allow"
      Action   = ["es:ESHttpGet", "es:ESHttpPost", "es:ESHttpPut"]
      Resource = "${aws_opensearch_domain.vectors[0].arn}/*"
    }
  ] : []

  lambda_statements = concat(local.base_statements, local.opensearch_statements)
}

resource "aws_iam_role_policy" "lambda" {
  name = "${var.project_name}-lambda-policy"
  role = aws_iam_role.lambda.id

  policy = jsonencode({
    Version   = "2012-10-17"
    Statement = local.lambda_statements
  })
}

resource "aws_lambda_function" "ingestion" {
  function_name = var.lambda_function_name
  role          = aws_iam_role.lambda.arn
  handler       = var.lambda_handler
  runtime       = var.lambda_runtime
  timeout       = var.lambda_timeout
  memory_size   = var.lambda_memory_size

  filename         = var.lambda_zip_path
  source_code_hash = fileexists(var.lambda_zip_path) ? filebase64sha256(var.lambda_zip_path) : null

  environment {
    variables = {
      LLM_PROVIDER               = "bedrock"
      EMBEDDING_PROVIDER         = "bedrock"
      VECTOR_BACKEND             = var.enable_opensearch ? "opensearch" : "chroma"
      STORAGE_BACKEND            = "s3"
      MEMORY_BACKEND             = "dynamodb"
      AWS_ENDPOINT_URL           = var.aws_endpoint_url
      S3_BUCKET                  = aws_s3_bucket.documents.id
      DYNAMODB_TABLE             = aws_dynamodb_table.state.name
      BEDROCK_MODEL_ID           = var.bedrock_model_id
      BEDROCK_EMBEDDING_MODEL_ID = var.bedrock_embedding_model_id
      OPENSEARCH_HOST            = var.enable_opensearch ? aws_opensearch_domain.vectors[0].endpoint : ""
    }
  }

  tags = var.tags
}

# Trigger ingestion automatically when an object lands in the bucket.
resource "aws_lambda_permission" "allow_s3" {
  statement_id  = "AllowExecutionFromS3"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.ingestion.function_name
  principal     = "s3.amazonaws.com"
  source_arn    = aws_s3_bucket.documents.arn
}

resource "aws_s3_bucket_notification" "ingest" {
  bucket = aws_s3_bucket.documents.id

  lambda_function {
    lambda_function_arn = aws_lambda_function.ingestion.arn
    events              = ["s3:ObjectCreated:*"]
  }

  depends_on = [aws_lambda_permission.allow_s3]
}
