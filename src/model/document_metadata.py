from dataclasses import dataclass

from src.utils.document_processor import DocumentProcessor


@dataclass
class DocumentMetadata:
    document_processor: DocumentProcessor

    def set_document_processor(self, document_processor):
        self.document_processor = document_processor

    def get_document_processor(self):
        return self.document_processor
