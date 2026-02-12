variable "project_id" {}
variable "zone" {}

resource "google_compute_instance" "ml_box" {
  name         = "ml-gpu-box"
  machine_type = "g2-standard-32" 
  zone         = var.zone

  tags = ["ml-box"]

  boot_disk {
    initialize_params {
      # 2026-ready Deep Learning Image with pre-installed drivers
      image = "projects/deeplearning-platform-release/global/images/family/pytorch-2-7-cu128-ubuntu-2204-nvidia-570"
      size  = 100
    }
  }

  network_interface {
    network  = "default"
    nic_type = "GVNIC"
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
    startup-script = file("${path.module}/setup_ml_box.sh")
  }

  scheduling {
    on_host_maintenance = "TERMINATE" # Required for GPU instances
  }
}

output "ml_box_public_ip" {
  value = google_compute_instance.ml_box.network_interface[0].access_config[0].nat_ip
}

