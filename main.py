import os
import subprocess
from string import Template

def generate_service_file(service_name, description, user, source_directory, exec_start):
    template_path = os.path.join(os.path.dirname(__file__), "templates", "service_template.service")

    # Load the template file
    with open(template_path, "r") as f:
        template_content = Template(f.read())

    # Replace placeholders with actual values
    service_content = template_content.substitute(
        description=description,
        user=user,
        source_directory=source_directory,
        exec_start=exec_start
    )

    # Choose where to place the file based on OS
    output_path = f"/etc/systemd/system/{service_name}.service"

    # Write the final service file
    with open(output_path, "w") as f:
        f.write(service_content)

    print(f"Service file generated at: {output_path}")
    return output_path

def install_service(service_name):
    subprocess.run(["systemctl", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "enable", service_name], check=True)
    subprocess.run(["systemctl", "start", service_name], check=True)
    print(f"Service '{service_name}' installed and started.")


if __name__ == "__main__":
    services = [
        {
            "name": "ingestion_service",
            "description": "Ingestion Service",
            "script": "ingestion_service/main.py"
        },
        {
            "name": "orchestration_service",
            "description": "Orchestration Service",
            "script": "orchestration_service/main.py"
        }
    ]

    for svc in services:
        working_dir = os.path.dirname(__file__)
        source_dir = os.path.join(os.path.dirname(__file__), "src")

        exec_command = f"python3 {os.path.join(source_dir, svc['script'])}"

        path = generate_service_file(
            service_name=svc["name"],
            description=svc["description"],
            user=os.getenv("USER"),
            source_directory=source_dir,
            exec_start=exec_command
        )

        install_service(svc["name"])