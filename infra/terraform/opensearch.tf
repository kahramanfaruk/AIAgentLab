# COST WARNING
# A managed OpenSearch domain bills per hour for as long as it exists, even when
# idle. A single t3.small.search node is roughly 25 USD per month plus storage,
# and OpenSearch Serverless has a far higher minimum (about 345 USD per month).
# To protect the 200 USD credit budget this domain is disabled by default. During
# development the open-source OpenSearch Docker image (see docker-compose.yml)
# provides the same k-NN API at no cost. Enable this only deliberately:
#   terraform apply -var="enable_opensearch=true"

resource "aws_opensearch_domain" "vectors" {
  count          = var.enable_opensearch ? 1 : 0
  domain_name    = "${var.project_name}-vectors"
  engine_version = "OpenSearch_2.11"

  cluster_config {
    instance_type  = var.opensearch_instance_type
    instance_count = 1
  }

  ebs_options {
    ebs_enabled = true
    volume_size = var.opensearch_volume_size
  }

  node_to_node_encryption {
    enabled = true
  }

  encrypt_at_rest {
    enabled = true
  }

  domain_endpoint_options {
    enforce_https       = true
    tls_security_policy = "Policy-Min-TLS-1-2-2019-07"
  }

  tags = var.tags
}
