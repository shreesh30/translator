import logging
import os
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from docx import Document

from src.service.document_builder import DocumentBuilder
from src.service.rabbitmq_consumer import RabbitMQConsumer
from src.utils.utils import Utils

logger = logging.getLogger('result_handler')
class ResultHandler:
    def __init__(self):
        self.documents = {}
        self.lock = threading.Lock()  # protect shared dict
        self.executor = ThreadPoolExecutor(max_workers=4)

    def handle_complete_document(self, doc_id):
        """Called in a worker thread when all chunks for a doc are collected."""
        with self.lock:
            results = self.documents.pop(doc_id, [])

        if not results:
            return

        # Sort chunks by index
        results.sort(key=lambda r: r.chunk_index)

        logger.info(
            f"[GPUWorker] Finalizing document {doc_id} with {len(results)} chunks "
            f"(thread: {threading.current_thread().name})"
        )
        try:
            language_config= results[-1].language_config
            meta_data = results[-1].meta_data
            document_processor = meta_data.document_processor
            elements = [result.element for result in results]
            file_name = results[-1].filename
            language = language_config.target_language

            # Build the translated document
            doc = Document()
            builder = DocumentBuilder(
                document=doc,
                language_config=language_config,
                document_processor=document_processor
            )
            builder.build_document(elements)

            target_dir = os.path.join(Utils.OUTPUT_DIR, language)
            os.makedirs(target_dir, exist_ok=True)

            docx_filename = f"{Path(file_name).stem}.docx"
            save_path = os.path.join(str(target_dir), docx_filename)

            doc.save(str(save_path))
            logger.info(f"Saved {language} version of {file_name}")
        except Exception as e:
            logger.error(f"[GPUWorker] Error writing document {doc_id}: {e}", exc_info=True)

    def process_message(self,ch, method, properties, body):
        try:
            result = pickle.loads(body)
            logging.info(f"[Consumer] Received: {result}")
            logger.info(f"[GPUWorker] Received result {result.id}, chunk {result.chunk_index + 1}/{result.total_chunks}")

            with self.lock:
                if result.id not in self.documents:
                    self.documents[result.id] = [result]
                else:
                    self.documents[result.id].append(result)

                # Check if document is complete
                if len(self.documents[result.id]) == result.total_chunks:
                    logger.info(f"[GPUWorker] All chunks received for {result.id}")
                    # Submit processing to thread pool
                    self.executor.submit(self.handle_complete_document, result.id)

            # Acknowledge after processing
            ch.basic_ack(delivery_tag=method.delivery_tag)
            logging.info("[Consumer] Message acknowledged")
        except Exception as e:
            logger.error(f"[GPUWorker] Error: {e}")
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

    def run(self):
        consumer = RabbitMQConsumer(host=Utils.KEY_RABBITMQ_LOCALHOST, queue=Utils.QUEUE_RESULTS)

        try:
            consumer.connect()
            consumer.consume(callback=self.process_message)
        finally:
            consumer.close()