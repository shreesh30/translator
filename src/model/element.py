from dataclasses import dataclass, field
from typing import Optional
import fitz

@dataclass
class Element:
    page_number: int = field(default_factory=int)
    bbox: fitz.Rect = field(default_factory=fitz.Rect)  # bounding box of the element
    # start: float = field(default_factory=float)         # optional: start position (y)
    # end: float = field(default_factory=float)           # optional: end position (y)
    element_type: Optional[str] = None                  # e.g., "paragraph", "table", etc.

    def get_element_type(self):
        return self.element_type