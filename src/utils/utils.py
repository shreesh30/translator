import logging
import os
import subprocess
from logging.handlers import RotatingFileHandler
from string import Template


class Utils:
    TAGS = {'[123]': '<b>', '[456]': '</b>'}
    STANDARDIZED_PAGE_HEIGHT = 11.69
    STANDARDIZED_PAGE_WIDTH = 8.27
    STANDARDIZED_TOP_MARGIN = 1.48
    STANDARDIZED_BOTTOM_MARGIN = 1.38
    STANDARDIZED_LEFT_MARGIN = 1.38
    STANDARDIZED_RIGHT_MARGIN = 1.38
    STANDARDIZED_FOOTER_DISTANCE = 1
    USABLE_PAGE_HEIGHT = (STANDARDIZED_PAGE_HEIGHT - STANDARDIZED_TOP_MARGIN - STANDARDIZED_BOTTOM_MARGIN)
    USABLE_PAGE_WIDTH = (STANDARDIZED_PAGE_WIDTH - STANDARDIZED_LEFT_MARGIN - STANDARDIZED_RIGHT_MARGIN)
    TYPE_PARAGRAPH = "paragraph"
    TYPE_TABLE = "table"

    # RABBITMQ USER & PASSWORD
    KEY_USER = 'admin'
    KEY_PASSWORD = 'admin'
    KEY_RABBITMQ_HOST = '172.31.24.147'
    KEY_RABBITMQ_LOCALHOST = 'localhost'

    # RABBITMQ Keys
    QUEUE_TASKS = 'tasks'
    QUEUE_RESULTS = 'results'

    # SERVICES
    INGESTION_SERVICE = 'ingestion_service'
    TRANSLATION_SERVICE = 'translation_service'
    BUILDER_SERVICE = 'builder_service'

    # DIRECTORY
    LOG_DIR = 'logs'

    @staticmethod
    def setup_logging(log_file_name: str, max_bytes=10 * 1024 * 1024, backup_count=5):
        """Configure logging with rotation based on file size."""
        os.makedirs(Utils.LOG_DIR, exist_ok=True)
        log_path = os.path.join(Utils.LOG_DIR, log_file_name)

        handler = RotatingFileHandler(
            log_path, maxBytes=max_bytes, backupCount=backup_count
        )
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(processName)s - %(message)s"
        )
        handler.setFormatter(formatter)

        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        logger.handlers = []  # clear existing handlers
        logger.addHandler(handler)

    @staticmethod
    def generate_service_file(service_name, description, user, working_directory, exec_start):
        template_path = os.path.join( "src", "templates", "service_template.service")

        # Load the template file
        with open(template_path, "r") as f:
            template_content = Template(f.read())

        # Replace placeholders with actual values
        service_content = template_content.substitute(
            description=description,
            user=user,
            working_directory=working_directory,
            exec_start=exec_start
        )

        # Choose where to place the file based on OS
        output_path = f"/etc/systemd/system/{service_name}.service"

        # Write the final service file
        with open(output_path, "w") as f:
            f.write(service_content)

        print(f"Service file generated at: {output_path}")
        return output_path

    @staticmethod
    def install_service(service_name):
        subprocess.run(["systemctl", "daemon-reload"], check=True)
        subprocess.run(["systemctl", "enable", service_name], check=True)
        subprocess.run(["systemctl", "start", service_name], check=True)
        print(f"Service '{service_name}' installed and started.")