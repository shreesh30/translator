import gzip
import json
import logging
import os
import pickle
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from dacite import from_dict
from docx import Document

from src.model.result import Result
from src.utils.document_builder import DocumentBuilder
from src.utils.rabbitmq_consumer import RabbitMQConsumer
from src.utils.utils import Utils

logger = logging.getLogger(Utils.BUILDER_SERVICE)

class ResultHandler:
    def __init__(self):
        self.documents = {}
        self.lock = threading.Lock()  # protect shared dict
        self.executor = ThreadPoolExecutor(max_workers=2)

        self.cache_dir = Path(Utils.CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # On startup, rebuild from disk
        self._rebuild_from_disk()

    def _rebuild_from_disk(self):
        """On startup, reload incomplete docs from disk"""
        for doc_dir in self.cache_dir.iterdir():
            if not doc_dir.is_dir():
                continue
            doc_id = doc_dir.name
            chunks = []
            for chunk_file in sorted(doc_dir.glob("chunk_*.pkl")):
                with open(chunk_file, "rb") as f:
                    result = pickle.load(f)
                    chunks.append(result)
            if chunks:
                self.documents[doc_id] = chunks
                logger.info(f"[ResultHandler] Rebuilt {len(chunks)} chunks for doc {doc_id}")

    def _chunk_path(self, doc_id, chunk_index):
        doc_dir = self.cache_dir / str(doc_id)
        doc_dir.mkdir(parents=True, exist_ok=True)
        return doc_dir / f"chunk_{chunk_index}.pkl"

    def _persist_chunk(self, result: Result):
        """Persist each chunk to disk so it's restart-safe"""
        path = self._chunk_path(result.id, result.chunk_index)
        with open(path, "wb") as f:
            pickle.dump(result, f)

    def handle_complete_document(self, results):
        """Called in a worker thread when all chunks for a doc are collected."""
        if not results:
            return

        # Sort chunks by index
        results.sort(key=lambda r: r.chunk_index)

        doc_id = results[-1].id

        logger.info(
            f"[ResultHandler] Finalizing document {doc_id} with {len(results)} chunks "
            f"(thread: {threading.current_thread().name})"
        )
        try:
            language_config= results[-1].language_config
            meta_data = results[-1].meta_data
            elements = [result.element for result in results]
            file_name = results[-1].filename
            language = language_config.target_language

            # Build the translated document
            doc = Document()
            builder = DocumentBuilder(
                document=doc,
                language_config=language_config,
                meta_data=meta_data
            )
            builder.build_document(elements)

            target_dir = os.path.join(Utils.OUTPUT_DIR, language)
            os.makedirs(target_dir, exist_ok=True)

            docx_filename = f"{Path(file_name).stem}.docx"
            save_path = os.path.join(str(target_dir), docx_filename)

            doc.save(str(save_path))
            logger.info(f"Saved {language} version of {file_name}")

            # If done successfully, cleanup from disk

            doc_dir = self.cache_dir / str(results[-1].id)
            if doc_dir.exists():
                shutil.rmtree(doc_dir, ignore_errors=True)

            logger.info(f"[ResultHandler] Cleaned up temporary files for document {doc_id}")
        except Exception as e:
            logger.error(f"[ResultHandler] Error writing document {doc_id}: {e}", exc_info=True)

    def process_message(self,ch, method, properties, body):
        try:
            body_decompressed = gzip.decompress(body)
            result = pickle.loads(body_decompressed)

            if not isinstance(result, Result):
                raise TypeError(f"Expected Result, got {type(result)}")

            logger.info(f"[Consumer] Received result {result.id}, chunk {result.chunk_index + 1}/{result.total_chunks}")

            # Persist chunk
            self._persist_chunk(result)

            with self.lock:
                if result.id not in self.documents:
                    self.documents[result.id] = [result]
                else:
                    self.documents[result.id].append(result)

                ch.basic_ack(delivery_tag=method.delivery_tag)
                logging.info("[Consumer] Message acknowledged")

                # Check if document is complete
                if len(self.documents[result.id]) == result.total_chunks:
                    logger.info(f"[ResultHandler] All chunks received for {result.id}")
                    results = self.documents.pop(result.id)
                    # Submit processing to thread pool
                    # self.executor.submit(self.handle_complete_document, results)
                    self.handle_complete_document(results)
        except Exception as e:
            logger.error(f"[ResultHandler] Error: {e}", exc_info=True)
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

    def _worker(self, worker_id):
        consumer = RabbitMQConsumer(
            host=Utils.KEY_RABBITMQ_LOCALHOST,
            queue=Utils.QUEUE_RESULTS
        )
        try:
            consumer.connect()
            logging.info(f"[ResultHandler-{worker_id}] Consumer started")
            consumer.consume(callback=self.process_message)
        finally:
            consumer.close()

    def run(self):
        """Start multiple consumer threads"""
        threads = []
        for i in range(2):
            t = threading.Thread(target=self._worker, args=(i,), daemon=True)
            t.start()
            threads.append(t)
            logging.info(f"[ResultHandler] Started worker thread {i}")

        # Keep the main thread alive
        for t in threads:
            t.join()