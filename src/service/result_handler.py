import logging
import os
import queue
import threading

from docx import Document

from src.model.task_result import TaskResult
from src.service.document_builder import DocumentBuilder

logger = logging.getLogger('result_handler')
class ResultHandler:
    def __init__(self, output_queue, output_path):
        self.output_queue = output_queue
        self.output_path = output_path
        self._stop_event = threading.Event()

    def start(self):
        t = threading.Thread(target=self.run, daemon=True)
        t.start()

    def stop(self):
        self._stop_event.set()

    def run(self):
        while not self._stop_event.is_set():
            try:
                result: TaskResult = self.output_queue.get()

                logger.info(f'Task Result: {result}')

                # Handle error case
                if result.error is not None:
                    logger.error(f"Translation failed for {result.filename}: {result.error}")
                    continue

                # Build the translated document
                doc = Document()
                builder = DocumentBuilder(
                    document=doc,
                    language_config=result.language_config,
                    document_processor=result.document_processor
                )
                builder.build_document(result.elements)

                target_dir = os.path.join(self.output_path, result.language)
                os.makedirs(target_dir, exist_ok=True)

                docx_filename = f"{os.path.splitext(result.filename)[0]}.docx"
                save_path = os.path.join(target_dir, docx_filename)

                doc.save(save_path)
                logger.info(f"Saved {result.language} version of {result.filename}")

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in ResultHandler: {e}", exc_info=True)