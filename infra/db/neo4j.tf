# db/neo4j.tf

variable "project_id" {}
variable "zone" {}

resource "google_compute_instance" "neo4j_vm" {
  name         = "neo4j-gds-enterprise"
  machine_type = "n2-highmem-32" 
  zone         = var.zone

  # Required for the firewall rule to find this VM
  tags = ["neo4j"]

  network_performance_config {
    total_egress_bandwidth_tier = "TIER_1"
  }

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2204-lts"
      size  = 50
    }
  }

  # N2-highmem-32 requires 0, 4, 8, 16, or 24 local SSDs.
  # We will provision 4 to satisfy the requirement.
  scratch_disk {
    interface = "NVME"
  }
  scratch_disk {
    interface = "NVME"
  }
  scratch_disk {
    interface = "NVME"
  }
  scratch_disk {
    interface = "NVME"
  }

  network_interface {
    network = "default"
    access_config {
      network_tier = "PREMIUM"
    }
  }

  metadata = {
    # Replace with your ssh pub key
    ssh-keys = <<EOT
      harry.wik:ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAICI6lic6qcFzxtwfGfnwRidhLGofs6LDsF0hfC5fTuND harry.wik@neo4j.com
    EOT
    
    # Load the config file into metadata
    neo4j_config_content = file("${path.module}/neo4j.conf")
    
    # Pass the startup script
    startup-script = file("${path.module}/setup_neo4j.sh")
  }

  allow_stopping_for_update = true

  service_account {
    # Use the default compute service account or a custom one
    scopes = ["cloud-platform"]
  }
}

# SSH Rule: Open to everyone
resource "google_compute_firewall" "ssh_all" {
  name    = "allow-ssh-from-anywhere"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  # 0.0.0.0/0 means the entire internet
  source_ranges = ["0.0.0.0/0"] 
  target_tags   = ["neo4j"]
}

# Neo4j Rule: Restricted to specific IPs
# resource "google_compute_firewall" "neo4j_restricted" {
#   name    = "allow-neo4j-restricted"
#   network = "default"
# 
#   allow {
#     protocol = "tcp"
#     # 8491 is for Arrow
#     ports    = ["7474", "7473", "7687", "8491"]
#   }
# 
#   # Add your specific IP (e.g., "1.2.3.4/32") and 
#   # any other box's IP here. 
#   # Note: "localhost" access is handled internally by the OS, 
#   # but internal VPC traffic needs the VPC range here if used.
#   source_ranges = ["YOUR_OFFICE_OR_HOME_IP/32", "GPU_BOX_PRIVATE_IP/32"] 
#   target_tags   = ["neo4j"]
# }

output "external_ip" {
  value = google_compute_instance.neo4j_vm.network_interface[0].access_config[0].nat_ip
}
