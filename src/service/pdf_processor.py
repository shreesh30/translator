import json
import logging
import os
import uuid
from concurrent.futures import as_completed
from concurrent.futures.thread import ThreadPoolExecutor
from dataclasses import asdict

from src.model.document_metadata import DocumentMetadata
from src.model.task import Task
from src.service.document_processor import DocumentProcessor
from src.service.rabbitmq_producer import RabbitMQProducer
from src.utils.custom_encoder import CustomJSONEncoder
from src.utils.utils import Utils

logger = logging.getLogger(Utils.INGESTION_SERVICE)


class PDFProcessor:
    def __init__(self, lang_configs, input_path):
        self.lang_configs = lang_configs
        self.input_path = input_path

    def process_all_pdfs(self):
        pdf_files = [f for f in os.listdir(self.input_path) if f.endswith(".pdf")]

        with ThreadPoolExecutor(max_workers=8) as executor:  # adjust workers as needed
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
            file_path = os.path.join(self.input_path, filename)
            processor = DocumentProcessor(file_path)
            processor.process_document()
            elements = processor.get_elements()
            total_chunks = len(elements)
            extracted_page_number, page_number_start = processor.get_page_number_info()
            metadata= DocumentMetadata(paragraph_start=processor.get_paragraph_start(),pages=processor.get_pages(),extracted_page_number=extracted_page_number, page_number_start=page_number_start)

            for language_config in self.lang_configs:
                task_id = uuid.uuid4().hex
                for idx, element in enumerate(elements):
                    task = Task(id=task_id, element = element, language_config=language_config, filename=filename, chunk_index=idx, total_chunks=total_chunks, meta_data=metadata)
                    task_json = json.dumps(asdict(task), cls=CustomJSONEncoder) # type: ignore[arg-type]
                    producer.publish(task_json)
                    logger.info(f"Queued chunk {idx+1}/{total_chunks} for {filename} in {language_config.get_target_language()} (task_id={task_id})")

        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
        finally:
            producer.close()