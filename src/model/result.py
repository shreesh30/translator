from dataclasses import dataclass

from src.model.document_metadata import DocumentMetadata
from src.model.element import Element
from src.model.language_config import LanguageConfig


@dataclass
class Result:
    id: str
    element: object
    filename: str
    language_config: LanguageConfig
    chunk_index: int
    total_chunks: int
    meta_data: DocumentMetadata