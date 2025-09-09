import logging
import os
import shutil
import subprocess
from logging.handlers import RotatingFileHandler
from string import Template

from dacite import from_dict, Config

from src.model.bbox import Bbox
from src.model.footer import Footer
from src.model.line import Line
from src.model.paragraph import Paragraph
from src.model.table import Table


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
    OUTPUT_DIR = 'resource/output'
    # INPUT_DIR = 'resource/tmp'
    INPUT_DIR = "resource/input/pdf-complete"
    CACHE_DIR = '/var/cache/translator'


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
    def clear_directory(directory_name):
        if os.path.exists(directory_name):
            # Remove all contents inside the folder
            for filename in os.listdir(directory_name):
                file_path = os.path.join(directory_name, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)  # remove file or symlink
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)  # remove folder recursively
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")
        else:
            os.makedirs(directory_name)

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

    @staticmethod
    def terminate_service(service_name):
        # Stop and delete existing service if running
        try:
            subprocess.run(["sudo", "systemctl", "stop", service_name], check=False)
            subprocess.run(["sudo", "systemctl", "disable", service_name], check=False)
            # Remove service file if it exists
            service_file = f"/etc/systemd/system/{service_name}.service"
            if os.path.exists(service_file):
                os.remove(service_file)
        except Exception as e:
            print(f"Failed to stop/delete {service_name}: {e}")

    @staticmethod
    def get_cast_classes():
        cast_classes = [Paragraph, Table, Line, Footer, Bbox]
        return cast_classes

    @staticmethod
    def element_factory(data: dict):
        cast_classes = Utils.get_cast_classes()

        config = Config(
            cast=cast_classes,
            strict=False  # allow dicts to be converted into dataclasses
        )

        if data.get("type") == Utils.TYPE_PARAGRAPH:
            return from_dict(Paragraph, data, config=config)
        elif data.get("type") == Utils.TYPE_TABLE:
            return from_dict(Table, data, config=config)
        else:
            raise ValueError(f"Unknown element type: {data.get('type')}")

    @staticmethod
    def get_config():
        config = Config(
            cast=[Paragraph, Table, Line, Footer, Bbox],
            type_hooks={object: Utils.element_factory},
            strict=False
        )

        return config