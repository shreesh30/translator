import logging

from src.translation_service.gpu_worker import GPUWorker
from src.utils.utils import Utils


def run_gpu_worker():
    logger = logging.getLogger(Utils.TRANSLATION_SERVICE)

    # Start processing PDFs
    logger.info("Starting GPU Worker")
    worker = GPUWorker(
        model_name="ai4bharat/indictrans2-en-indic-1B",
        quantization="8-bit"
    )
    worker.run()


if __name__ == "__main__":
    Utils.setup_logging(f"{Utils.TRANSLATION_SERVICE}.log")
    run_gpu_worker()
