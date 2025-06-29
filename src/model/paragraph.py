from dataclasses import dataclass, field
from typing import List

import fitz

from src.model.footer import Footer
from src.model.line import Line


@dataclass
class Paragraph:
    page_number:int = field(default_factory=int, repr=False)
    lines: List[Line] = field(default_factory=list)
    footer: List[Footer] = field(default_factory=list)
    font_size: float = field(default_factory=float)
    para_bbox: fitz.Rect = field(default_factory=fitz.Rect)
    start: float = field(default_factory=float)
    end: float = field(default_factory=float)

    def set_lines(self, lines):
        self.lines = lines

    def set_font_size(self, font_size):
        self.font_size = font_size

    def set_para_bbox(self, para_bbox):
        self.para_bbox = para_bbox

    def set_footers(self):
        pass

    def set_start(self, start):
        self.start = start

    def set_end(self, end):
        self.end = end

    def get_lines(self):
        return self.lines

    def get_font_size(self):
        return self.font_size

    def get_page_number(self):
        return self.page_number

    def get_para_bbox(self):
        return self.para_bbox

    def get_footer(self):
        return self.footer

    def get_start(self):
        return self.start

    def get_end(self):
        return self.end