import logging
import os
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from typing import List

from docx import Document

from src.model.language_config import LanguageConfig
from src.model.paragraph import Paragraph
from src.model.line import Line
from src.model.table import Table
from src.service.document_builder import DocumentBuilder
from src.service.document_processor import DocumentProcessor

logger = logging.getLogger(__name__)


class PDFProcessor:
    def __init__(self, lang_config: LanguageConfig, gpu_task_queue: Queue,
                 gpu_result_queue: Queue, input_path: str, output_path: str):
        self.lang_config = lang_config
        self.gpu_task_queue = gpu_task_queue
        self.gpu_result_queue = gpu_result_queue
        self.input_path = input_path
        self.output_path = output_path

    def translate_text(self, text: str) -> str:
        """Send text to GPU worker and wait for result"""
        result_queue = Queue()
        self.gpu_task_queue.put({
            'text': text,
            'lang': self.lang_config.target_language_key,
            'result_queue': result_queue
        })
        return result_queue.get()

    def process_file(self, filename: str):
        try:
            file_path = os.path.join(self.input_path, filename)
            logger.info(f'Processing {file_path}')

            # Use existing DocumentProcessor
            doc_processor = DocumentProcessor(file_path, self.lang_config)
            doc_processor.process_document()

            # Translate elements
            self._translate_elements(doc_processor.get_elements())

            # Build DOCX
            self._build_docx(doc_processor, filename)

        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")

    def _translate_elements(self, elements):
        """Translate all paragraphs and tables"""
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Process paragraphs
            paragraphs = [e for e in elements if isinstance(e, Paragraph)]
            list(executor.map(self._translate_paragraph, paragraphs))

            # Process tables
            tables = [e for e in elements if isinstance(e, Table)]
            list(executor.map(self._translate_table, tables))

    def _translate_paragraph(self, paragraph: Paragraph):
        try:
            text = " ".join(line.get_text() for line in paragraph.get_lines())
            translated = self.translate_text(text)

            # Update paragraph with translated text
            lines = self._split_into_lines(translated, paragraph)
            paragraph.set_lines(lines)

            # Translate sub-paragraphs if any
            for sub_para in paragraph.get_sub_paragraphs():
                self._translate_paragraph(sub_para)

            # Translate footer if exists
            if paragraph.get_footer():
                for footer in paragraph.get_footer():
                    footer.text = self.translate_text(footer.text)

        except Exception as e:
            logger.error(f"Error translating paragraph: {e}")

    def _translate_table(self, table: Table):
        try:
            if table.get_title().get_text():
                table.get_title().set_text(self.translate_text(table.get_title().get_text()))

                if table.get_sub_title().get_text():
                    table.get_sub_title().set_text(self.translate_text(table.get_sub_title().get_text()))

                for column in table.get_columns():
                    for row in column:
                        row.set_text(self.translate_text(row.get_text()))
        except Exception as e:
            logger.error(f"Error translating table: {e}")

    def _split_into_lines(self, text: str, paragraph: Paragraph) -> List[Line]:
        """Adapt your existing line splitting logic here"""
        # Implement using your existing PDFTranslator._split_words method
        pass

    def _build_docx(self, doc_processor: DocumentProcessor, filename: str):
        doc = Document()
        builder = DocumentBuilder(document=doc,language_config= self.lang_config, document_processor=doc_processor)
        builder.build_document(doc_processor.get_elements())

        output_dir = os.path.join(
            self.output_path,
            self.lang_config.target_language
        )
        os.makedirs(output_dir, exist_ok=True)
        doc.save(os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.docx"))

    def run(self):
        """Process all PDFs for this language"""
        pdf_files = [
            f for f in os.listdir(self.input_path)
            if f.lower().endswith(".pdf")
        ]

        with ThreadPoolExecutor(max_workers=2) as executor:
            executor.map(self.process_file, pdf_files)