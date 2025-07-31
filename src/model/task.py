from dataclasses import dataclass
from typing import Any, List

from src.model.language_config import LanguageConfig
from src.service.document_processor import DocumentProcessor


@dataclass
class Task:
    elements: List[dict]
    language_configs: List[LanguageConfig]
    filename: str
    processor: DocumentProcessor