from dataclasses import dataclass, field
from typing import Tuple

import fitz

from src.model.bbox import Bbox


@dataclass
class Line:
    page_number: int = field(default_factory=int, repr=False)
    text: str = ""
    line_bbox: Bbox = field(default_factory=lambda: Bbox(0, 0, 0, 0))
    font_size: float = field(default_factory=int, repr=True)

    def set_text(self, text):
        self.text = text

    def set_line_bbox(self, bbox):
        self.line_bbox = bbox

    def set_font_size(self, font_size):
        self.font_size = font_size

    def get_text(self):
        return self.text

    def get_line_bbox(self):
        return self.line_bbox

    def get_page_number(self):
        return self.page_number

    def get_font_size(self):
        return self.font_size


@dataclass
class TableLine(Line):
    pass