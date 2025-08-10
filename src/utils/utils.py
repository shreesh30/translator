import logging
import os


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

    # RABBITMQ Keys
    QUEUE_TASKS = 'tasks'
    QUEUE_RESULTS = 'results'
    KEY_LOCALHOST = "localhost"

    @staticmethod
    def setup_logging(log_file_name: str):
        """Configure logging to a file for the current process."""
        os.makedirs("logs", exist_ok=True)
        logging.basicConfig(
            filename=os.path.join("logs", log_file_name),
            level=logging.DEBUG,
            format="%(asctime)s [%(levelname)s] %(processName)s - %(message)s",
            force=True  # override inherited loggers
        )