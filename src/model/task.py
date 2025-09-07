from dataclasses import dataclass

from src.model.document_metadata import DocumentMetadata
from src.model.element import Element
from src.model.language_config import LanguageConfig


@dataclass
class Task:
    id: str
    element: Element
    filename: str
    meta_data: DocumentMetadata
    language_config: LanguageConfig
    chunk_index: int
    total_chunks: int