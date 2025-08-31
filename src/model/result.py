from dataclasses import dataclass
from typing import List, Any, Optional

from src.model.document_metadata import DocumentMetadata
from src.model.element import Element
from src.model.language_config import LanguageConfig
from src.service.document_processor import DocumentProcessor


@dataclass
class Result:
    id: str
    element: Element
    filename: str
    language_config: LanguageConfig
    chunk_index: int
    total_chunks: int
    meta_data: DocumentMetadata