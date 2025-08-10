from dataclasses import dataclass
from typing import Any, List

from src.model.element import Element
from src.model.language_config import LanguageConfig
from src.service.document_processor import DocumentProcessor


@dataclass
class Task:
    id: str
    element: Element
    filename: str
    # processor: DocumentProcessor
    language_config: LanguageConfig
    chunk_index: int
    total_chunks: int