# main.tf

provider "google" {
  project = "data-science-python-runtime"
  region  = "europe-west4"
}

# Database Module
module "neo4j_instance" {
  source     = "./db"
  project_id = "data-science-python-runtime"
  zone       = "europe-west4-a"
}

# ML Module
module "ml_instance" {
  source     = "./ml"
  project_id = "data-science-python-runtime"
  zone       = "europe-west4-a"
}

# Outputs for easy access
output "neo4j_public_ip" {
  value = module.neo4j_instance.external_ip
}

output "ml_box_public_ip" {
  value = module.ml_instance.ml_box_public_ip
}

