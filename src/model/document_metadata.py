from dataclasses import dataclass, asdict
from typing import List, Dict, Any

from src.model.page import Page
from src.service.document_processor import DocumentProcessor


@dataclass
class DocumentMetadata:
    # paragraph_start: int
    # extracted_page_number: int
    # page_number_start: int
    # pages: List[Page]
    document_processor: DocumentProcessor

    # def set_paragraph_start(self, paragraph_start):
    #     self.paragraph_start = paragraph_start
    #
    # def get_paragraph_start(self):
    #     return  self.paragraph_start
    #
    # def set_extracted_page_number(self, page_number):
    #     self.extracted_page_number = page_number
    #
    # def get_extracted_page_number(self):
    #     return self.extracted_page_number
    #
    # def set_page_number_start(self, page_number):
    #     self.page_number_start = page_number
    #
    # def get_page_number_start(self):
    #     return self.page_number_start
    #
    # def set_pages(self, pages):
    #     self.pages = pages
    #
    # def get_pages(self):
    #     return self.pages

    def set_document_processor(self, document_processor):
        self.document_processor = document_processor

    def get_document_processor(self):
        return self.document_processor
