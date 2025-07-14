from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import fitz

from src.model.footer import Footer
from src.model.line import Line


@dataclass
class Paragraph:
    page_number:int = field(default_factory=int)
    lines: List[Line] = field(default_factory=list)
    footer: List[Footer] = field(default_factory=list)
    font_size: float = field(default_factory=float)
    para_bbox: fitz.Rect = field(default_factory=fitz.Rect)
    start: float = field(default_factory=float)
    end: float = field(default_factory=float)
    sub_paragraphs: List[Paragraph] = field(default_factory=list)
    chapter: str = field(default_factory=str)
    volume: str = field(default_factory=str)

    def set_lines(self, lines):
        self.lines = lines

    def set_font_size(self, font_size):
        self.font_size = font_size

    def set_para_bbox(self, para_bbox):
        self.para_bbox = para_bbox

    def add_footers(self, footer):
        self.footer.append(footer)

    def set_footers(self,footers):
        self.footer = footers

    def set_start(self, start):
        self.start = start

    def set_end(self, end):
        self.end = end

    def set_page_number(self, page_number):
        self.page_number = page_number

    def set_sub_paragraphs(self, sub_paragraphs):
        self.sub_paragraphs = sub_paragraphs

    def set_chapter(self, chapter):
        self.chapter = chapter

    def set_volume(self, volume):
        self.volume = volume

    def add_sub_paragraph(self,  sub_paragraph):
        self.sub_paragraphs.append(sub_paragraph)

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

    def get_sub_paragraphs(self):
        return self.sub_paragraphs

    def get_chapter(self):
        return self.chapter

    def get_volume(self):
        return self.volume