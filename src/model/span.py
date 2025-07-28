from dataclasses import dataclass, field

import fitz


@dataclass
class Span:
    text: str = field(default_factory=str)
    font: str = field(default_factory=str, repr = False)
    font_size: float = field(default_factory=float, repr=False)
    page_num: int = field(default_factory=float, repr=False)
    origin: tuple = field(default_factory=tuple, repr=True)
    bbox: fitz.Rect = field(default_factory=lambda: fitz.Rect(0, 0, 0, 0), repr=True)

    def set_text(self, text):
        self.text = text

    def get_text(self) -> str:
        return self.text

    def set_font(self, font):
        self.font = font

    def get_font(self):
        return self.font

    def get_font_size(self):
        return self.font_size

    def set_font_size(self,font_size):
        self.font_size = font_size

    def get_page_num(self):
        return self.page_num

    def set_page_num(self, page_num):
        self.page_num= page_num

    def get_origin(self):
        return self.origin

    def set_origin(self, origin):
        self.origin = origin

    def get_bbox(self):
        return self.bbox

    def set_bbox(self, bbox):
        self.bbox = bbox
