import logging
import os
import pickle
import uuid
from concurrent.futures import as_completed
from concurrent.futures.thread import ThreadPoolExecutor

from src.model.document_metadata import DocumentMetadata
from src.model.task import Task
from src.utils.document_processor import DocumentProcessor
from src.utils.rabbitmq_producer import RabbitMQProducer
from src.utils.utils import Utils

logger = logging.getLogger(Utils.INGESTION_SERVICE)


class PDFProcessor:
    def __init__(self, lang_configs):
        self.lang_configs = lang_configs

    def process_all_pdfs(self):
        pdf_files = [f for f in os.listdir(Utils.INPUT_DIR) if f.endswith(".pdf")]

        with ThreadPoolExecutor(max_workers=4) as executor:  # adjust workers as needed
            futures = [
                executor.submit(self.process_single_pdf, filename)
                for filename in pdf_files
            ]

            # Optional: Wait for completion & log
            for future in as_completed(futures):
                try:
                    future.result()  # raise if exception occurred
                except Exception as e:
                    logger.error(f"Error processing a file: {e}")

    def process_single_pdf(self, filename):
        producer = RabbitMQProducer(host=Utils.KEY_RABBITMQ_LOCALHOST, queue=Utils.QUEUE_TASKS)

        try:
            producer.connect()
            file_path = os.path.join(Utils.INPUT_DIR, filename)
            processor = DocumentProcessor(file_path)
            processor.process_document()
            elements = processor.get_elements()
            total_chunks = len(elements)
            metadata= DocumentMetadata(document_processor=processor)

            for language_config in self.lang_configs:
                task_id = uuid.uuid4().hex
                for idx, element in enumerate(elements):
                    task = Task(id=task_id, element = element, language_config=language_config, filename=filename, chunk_index=idx, total_chunks=total_chunks, meta_data=metadata)
                    task_body = pickle.dumps(task)
                    logger.info(f'Publishing Task Json: {task}')
                    producer.publish(task_body, persistent=False)
                    logger.info(f"Queued chunk {idx+1}/{total_chunks} for {filename} in {language_config.get_target_language()} (task_id={task_id})")
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
        finally:
            producer.close()