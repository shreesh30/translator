import logging

from src.builder_service.result_handler import ResultHandler
from src.utils.utils import Utils


def run_result_handler():
    logger = logging.getLogger(Utils.BUILDER_SERVICE)

    # Start processing PDFs
    logger.info("Starting Result Handler")
    handler = ResultHandler()
    handler.run()

if __name__ == '__main__':
    Utils.setup_logging(f"{Utils.BUILDER_SERVICE}.log")
    run_result_handler()
