output "s3_bucket_name" {
  description = "Name of the document bucket."
  value       = aws_s3_bucket.documents.id
}

output "dynamodb_table_name" {
  description = "Name of the state table."
  value       = aws_dynamodb_table.state.name
}

output "lambda_function_name" {
  description = "Name of the ingestion Lambda function."
  value       = aws_lambda_function.ingestion.function_name
}

output "opensearch_endpoint" {
  description = "OpenSearch domain endpoint, or null when disabled."
  value       = var.enable_opensearch ? aws_opensearch_domain.vectors[0].endpoint : null
}
