import logging
import os
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from typing import List
import threading

from docx import Document

from src.model.language_config import LanguageConfig
from src.model.paragraph import Paragraph
from src.model.line import Line
from src.model.table import Table
from src.service.document_builder import DocumentBuilder
from src.service.document_processor import DocumentProcessor
from IndicTransToolkit.processor import IndicProcessor

logger = logging.getLogger(__name__)


class PDFProcessor:
    def __init__(self,
                 lang_config: LanguageConfig,
                 gpu_task_queue: Queue,
                 gpu_result_queue: Queue,
                 input_path: str,
                 output_path: str,
                 max_workers: int = 4):
        self.lang_config = lang_config
        self.gpu_task_queue = gpu_task_queue
        self.gpu_result_queue = gpu_result_queue
        self.input_path = input_path
        self.output_path = output_path
        self.max_workers = max_workers
        self._lock = threading.Lock()

    def translate_text(self, text: str) -> str:
        """Thread-safe translation request"""
        result_queue = Queue()

        with self._lock:  # Protect queue operations
            self.gpu_task_queue.put({
                'text': text,
                'lang': self.lang_config.target_language_key,
                'result_queue': result_queue
            })
            return result_queue.get()

    def process_file(self, filename: str):
        """Process single PDF file"""
        try:
            file_path = os.path.join(self.input_path, filename)
            logger.info(f"Processing {file_path} for {self.lang_config.target_language}")

            # 1. Extract document structure
            doc_processor = DocumentProcessor(file_path, self.lang_config)
            doc_processor.process_document()
            elements = doc_processor.get_elements()

            # 2. Parallel translation
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all translation tasks
                futures = []
                for element in elements:
                    if element.type == "paragraph":
                        futures.append(executor.submit(
                            self._translate_paragraph,
                            element
                        ))
                    elif element.type == "table":
                        futures.append(executor.submit(
                            self._translate_table,
                            element
                        ))

                # Wait for completion
                for future in futures:
                    future.result()

            # 3. Build output document
            self._build_docx(elements, filename)

        except Exception as e:
            logger.error(f"Failed processing {filename}: {e}")

    def _translate_paragraph(self, paragraph):
        """Translate paragraph content"""
        try:
            # Combine all lines for context-aware translation
            full_text = " ".join(line.get_text() for line in paragraph.get_lines())
            translated = self.translate_text(full_text)

            # Reconstruct lines with original formatting
            lines = self._split_into_lines(translated, paragraph)
            paragraph.set_lines(lines)

            # Process sub-paragraphs if any
            for sub_para in paragraph.get_sub_paragraphs():
                self._translate_paragraph(sub_para)

            return paragraph

        except Exception as e:
            logger.error(f"Paragraph translation failed: {e}")
            return paragraph

    def _translate_table(self, table):
        """Translate table content"""
        try:
            # Translate title
            if table.get_title().get_text():
                translated = self.translate_text(table.get_title().get_text())
                table.get_title().set_text(translated)

            if table.get_sub_title().get_text():
                translated = self.translate_text(table.get_sub_title().get_text())
                table.get_sub_title().set_text(translated)

            # Translate content
            for column in table.get_columns():
                for row in column:
                    if row.get_text():
                        translated = self.translate_text(row.get_text())
                        row.set_text(translated)

            return table
        except Exception as e:
            logger.error(f"Table translation failed: {e}")
            return table

    def _split_into_lines(self, text: str, paragraph) -> List:
        """Adapt your existing line splitting logic"""
        # Implement based on your original PDFTranslator._split_words()
        pass

    def _build_docx(self, elements, filename):
        """Generate translated DOCX"""
        doc = Document()
        builder = DocumentBuilder(doc, self.lang_config)

        try:
            for element in elements:
                if element.type == "paragraph":
                    builder.add_paragraph(element)
                elif element.type == "table":
                    builder.add_table(element)

            output_dir = os.path.join(
                self.output_path,
                self.lang_config.target_language
            )
            os.makedirs(output_dir, exist_ok=True)

            output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.docx")
            doc.save(output_path)
            logger.info(f"Saved translation to {output_path}")

        except Exception as e:
            logger.error(f"DOCX generation failed: {e}")

    def run(self):
        """Process all PDFs for this language"""
        pdf_files = [
            f for f in os.listdir(self.input_path)
            if f.lower().endswith('.pdf')
        ]

        with ThreadPoolExecutor(max_workers=2) as executor:  # 2 files at a time
            executor.map(self.process_file, pdf_files)