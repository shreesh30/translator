from dataclasses import dataclass, field
from typing import Tuple

import fitz


@dataclass
class Line:
    page_number: int = field(default_factory=int, repr=False)
    text: str = ""
    line_bbox: fitz.Rect = field(default_factory=lambda: fitz.Rect(0, 0, 0, 0))
    origin: Tuple[float, float]  = field(default_factory=tuple)
    font_size: float = field(default_factory=int, repr=True)

    def set_text(self, text):
        self.text = text

    def set_line_bbox(self, bbox):
        self.line_bbox = bbox

    def set_origin(self, origin):
        self.origin =  origin

    def set_font_size(self, font_size):
        self.font_size = font_size

    def get_text(self):
        return self. text

    def get_line_bbox(self):
        return self.line_bbox

    def get_origin(self):
        return self.origin

    def get_page_number(self):
        return self.page_number

    def get_font_size(self):
        return self.font_size


@dataclass
class TableLine(Line):
    pass