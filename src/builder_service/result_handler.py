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
        """On startup, reload incomplete docs from disk using paths only"""
        logger.info(f"[ResultHandler] Rebuilding state from disk: {self.cache_dir}")

        if not self.cache_dir.exists():
            logger.warning(f"[ResultHandler] Cache dir {self.cache_dir} does not exist. Nothing to rebuild.")
            return

        for doc_dir in self.cache_dir.iterdir():
            if not doc_dir.is_dir():
                logger.debug(f"[ResultHandler] Skipping non-directory entry: {doc_dir}")
                continue

            doc_id = doc_dir.name
            logger.info(f"[ResultHandler] Processing cached document {doc_id}")

            chunk_paths = []
            for chunk_file in sorted(doc_dir.glob("chunk_*.pkl")):
                if chunk_file.is_file():
                    chunk_paths.append(str(chunk_file))
                    logger.debug(f"[ResultHandler] Found chunk {chunk_file.name} for doc {doc_id}")
                else:
                    logger.warning(f"[ResultHandler] Skipping invalid entry {chunk_file} in {doc_dir}")

            if chunk_paths:
                self.documents[doc_id] = chunk_paths
                logger.info(f"[ResultHandler] Rebuilt {len(chunk_paths)} chunks for doc {doc_id}")
            else:
                logger.warning(f"[ResultHandler] No valid chunks found for doc {doc_id}")

    def _chunk_path(self, doc_id, chunk_index):
        doc_dir = self.cache_dir / str(doc_id)
        doc_dir.mkdir(parents=True, exist_ok=True)
        return doc_dir / f"chunk_{chunk_index}.pkl"

    def _persist_chunk(self, result: Result):
        """Persist each chunk to disk so it's restart-safe"""
        path = self._chunk_path(result.id, result.chunk_index)
        with open(path, "wb") as f:
            pickle.dump(result, f)
        return str(path)

    def handle_complete_document(self, doc_id, chunk_paths):
        """Called in a worker thread when all chunks for a doc are collected."""
        if not chunk_paths:
            return

        logger.info(
            f"[ResultHandler] Finalizing document {doc_id} with {len(chunk_paths)} chunks "
            f"(thread: {threading.current_thread().name})"
        )

        try:
            # Sort chunk paths by index
            sorted_paths = sorted(chunk_paths, key=lambda p: int(Path(p).stem.split("_")[1]))

            language_config = None
            meta_data = None
            file_name = None

            # Create empty doc + builder
            doc = Document()
            builder = None

            for chunk_file in sorted_paths:
                with open(chunk_file, "rb") as f:
                    result = pickle.load(f)

                if language_config is None:  # only set once
                    language_config = result.language_config
                    meta_data = result.meta_data
                    file_name = result.filename
                    builder = DocumentBuilder(
                        document=doc,
                        language_config=language_config,
                        meta_data=meta_data
                    )

                # Add element directly (don’t keep in memory)
                builder.build_document([result.element])

            if builder is not None:
                builder.add_page_numbers()

            language = language_config.target_language
            target_dir = os.path.join(Utils.OUTPUT_DIR, language)
            os.makedirs(target_dir, exist_ok=True)

            docx_filename = f"{Path(file_name).stem}.docx"
            save_path = os.path.join(str(target_dir), docx_filename)

            doc.save(str(save_path))
            logger.info(f"Saved {language} version of {file_name}")

            # Cleanup temp files
            doc_dir = self.cache_dir / str(doc_id)
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
            chunk_path = self._persist_chunk(result)

            with self.lock:
                if result.id not in self.documents:
                    self.documents[result.id] = [chunk_path]
                else:
                    self.documents[result.id].append(chunk_path)

                # Check if document is complete
                if len(self.documents[result.id]) == result.total_chunks:
                    logger.info(f"[ResultHandler] All chunks received for {result.id}")
                    chunk_paths = self.documents.pop(result.id)
                    # Submit processing to thread pool
                    # self.executor.submit(self.handle_complete_document, results)
                    self.handle_complete_document(result.id, chunk_paths)

                # Acknowledge while still holding the lock
                ch.basic_ack(delivery_tag=method.delivery_tag)
                logging.info("[Consumer] Message acknowledged")
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
        for i in range(4):
            t = threading.Thread(target=self._worker, args=(i,), daemon=True)
            t.start()
            threads.append(t)
            logging.info(f"[ResultHandler] Started worker thread {i}")

        # Keep the main thread alive
        for t in threads:
            t.join()