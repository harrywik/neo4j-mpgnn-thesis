# main.tf

provider "google" {
  project = "data-science-python-runtime"
  region  = "europe-west4"
}

# Call the Neo4j configuration from the db folder
module "neo4j_instance" {
  source = "./db"
  
  # Pass variables here if you want to make the module reusable
  project_id = "data-science-python-runtime"
  zone       = "europe-west4-a"
}

# Output the IP so you know where to SSH or SCP
output "neo4j_public_ip" {
  value = module.neo4j_instance.external_ip
}
