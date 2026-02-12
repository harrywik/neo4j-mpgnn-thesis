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
      victor.pekkari:ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQCozJpm4f2hG/icj4I06H0HYlo1pnFGQqb+3ax+0YgIxwXxXNS7YQ99MGI6whm1mjRLhgYY+IGhPqb5ivlPV13XCP9AhTHflqL9v3IPM8N45oxj8SqAp+96uuoGDsUV9ZBjr8Hz1RaUTauiO5c5eAEJ58DW0NVKT/VM+0yXIVrGQ6242YhuqEGr0VrpwusYMlaEtPw20RwZJHDUXg09LtMMscv9lYrkNHxU+4Vpzr7OU7zw29zU/r1RjuK3n1CGunKehFCLuIqK6Y6rW7mc3ZoCdLHFfh7uVxELzoPopr8NJ8+Nw/ixpSZmwgRVyCdnUAlrT8qrYIr0y48MLPkQ/He4WHFRjyiS3181AWBbORBfHC5v5ken+Xqeu9/m7C61QtInIQ386/qNhwtD2K8DmBvkwLxsnxTCAOjaxW6EfD396paQ2L1Z2/Lg+gipLoZpgq2yRUcE3KWcF+l2qaIHR1sKUiriZyBYkTcwrMQ9ANTh5v+20NqvpXl/Am+53VS4c4k= victor.pekkari@neo4j.com
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

resource "google_compute_firewall" "allow_ml_to_neo4j" {
  name    = "allow-ml-to-neo4j"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["7687", "8491"] 
  }

  source_tags = ["ml-box"]   # Targets the ML instance
  target_tags = ["neo4j"]    # Targets the Neo4j instance
}

output "external_ip" {
  value = google_compute_instance.neo4j_vm.network_interface[0].access_config[0].nat_ip
}
