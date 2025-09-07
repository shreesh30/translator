import argparse
import os

from src.utils.utils import Utils

log_dir = Utils.LOG_DIR
if __name__ == "__main__":
    # Remove all the logs
    Utils.clear_directory(Utils.LOG_DIR)

    # Clear Output Folder
    Utils.clear_directory(Utils.OUTPUT_DIR)

    parser = argparse.ArgumentParser(description="Setup services")
    parser.add_argument(
        "service",
        nargs="?",  # makes the argument optional
        choices=[Utils.TRANSLATION_SERVICE],
        help="Service to install (only used for translation_service)"
    )
    args = parser.parse_args()

    all_services = [
        {
            "name": Utils.INGESTION_SERVICE,
            "description": "Ingestion Service",
            "script": "ingestion_service/main.py"
        },
        {
            "name": Utils.TRANSLATION_SERVICE,
            "description": "Translation Service",
            "script": "translation_service/main.py"
        },
        {
            "name": Utils.BUILDER_SERVICE,
            "description": "Builder Service",
            "script": "builder_service/main.py"
        },
    ]

    if args.service is None:
        services = [svc for svc in all_services if svc["name"] != Utils.TRANSLATION_SERVICE]
    else:
        # Only translation_service if passed
        services = [svc for svc in all_services if svc["name"] == args.service]

    working_dir = os.path.dirname(__file__)
    source_dir = os.path.join(working_dir, "src")

    for svc in services:
        # --- Stop and delete existing service if running ---
        Utils.terminate_service(svc["name"])

        exec_command = f"{working_dir}/venv/bin/python3 {os.path.join(source_dir, svc['script'])}"

        path = Utils.generate_service_file(
            service_name=svc["name"],
            description=svc["description"],
            user=os.getenv("USER"),
            working_directory=working_dir,
            exec_start=exec_command
        )

        Utils.install_service(svc["name"])