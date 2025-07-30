from dataclasses import dataclass, field
from typing import Optional
import fitz

@dataclass
class Element:
    page_number: int = field(default_factory=int)
    # bbox: fitz.Rect = field(default_factory=fitz.Rect)  # bounding box of the element
    # start: float = field(default_factory=float)         # optional: start position (y)
    # end: float = field(default_factory=float)           # optional: end position (y)
    type: Optional[str] = None                  # e.g., "paragraph", "table", etc.
    font_size: float = field(default_factory=float)

    def get_type(self):
        return self.type

    def get_page_number(self):
        return self.page_number

    def get_font_size(self):
        pass

    def set_volume(self, volume_name):
        pass

    def set_chapter(self, chapter_name):
        pass

    def set_font_size(self, font_size):
        pass