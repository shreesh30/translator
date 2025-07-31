from dataclasses import dataclass
from typing import List, Any, Optional

from src.service.document_processor import DocumentProcessor


@dataclass
class TaskResult:
    elements: List[Any]
    language: str
    language_config: Any
    filename: str
    document_processor: Optional[DocumentProcessor] = None
    error: Optional[str] = None