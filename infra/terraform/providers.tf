terraform {
  required_version = ">= 1.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# The provider targets real AWS by default. Setting aws_endpoint_url (for example
# http://localhost:4566) points every service at LocalStack and relaxes the
# credential and account checks, so the same configuration drives free local
# integration testing and a real deployment.
provider "aws" {
  region = var.aws_region

  skip_credentials_validation = var.aws_endpoint_url != ""
  skip_metadata_api_check     = var.aws_endpoint_url != ""
  skip_requesting_account_id  = var.aws_endpoint_url != ""
  s3_use_path_style           = var.aws_endpoint_url != ""

  dynamic "endpoints" {
    for_each = var.aws_endpoint_url != "" ? [var.aws_endpoint_url] : []
    content {
      s3         = endpoints.value
      dynamodb   = endpoints.value
      lambda     = endpoints.value
      iam        = endpoints.value
      sts        = endpoints.value
      opensearch = endpoints.value
      sagemaker  = endpoints.value
    }
  }
}
