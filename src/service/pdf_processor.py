import logging
import os
from concurrent.futures import as_completed
from concurrent.futures.thread import ThreadPoolExecutor

from IndicTransToolkit.processor import IndicProcessor

from src.model.task import Task
from src.service.document_processor import DocumentProcessor

logger = logging.getLogger(__name__)


class PDFProcessor:
    def __init__(self, lang_configs, input_path, gpu_task_queue):
        self.lang_configs = lang_configs
        self.input_path = input_path
        self.gpu_task_queue = gpu_task_queue

    def process_all_pdfs(self):
        pdf_files = [f for f in os.listdir(self.input_path) if f.endswith(".pdf")]

        with ThreadPoolExecutor(max_workers=8) as executor:  # adjust workers as needed
            futures = [
                executor.submit(self.process_single_pdf, filename, self.input_path, self.gpu_task_queue)
                for filename in pdf_files
            ]

            # Optional: Wait for completion & log
            for future in as_completed(futures):
                try:
                    future.result()  # raise if exception occurred
                except Exception as e:
                    logger.error(f"Error processing a file: {e}")

    def process_single_pdf(self, filename, input_path, input_queue):
        try:
            file_path = os.path.join(input_path, filename)
            processor = DocumentProcessor(file_path)
            processor.process_document()
            elements = processor.get_elements()

            task = Task(elements=elements, language_configs=self.lang_configs, filename=filename, processor=processor)
            input_queue.put(task)

        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")