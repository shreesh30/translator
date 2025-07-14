from dataclasses import field, dataclass

import fitz


@dataclass
class Drawing:
    page_number: int = field(default_factory=int)
    bbox:fitz.Rect = field(default_factory=fitz.Rect)

    def set_page_number(self, page_number):
        self.page_number = page_number

    def get_page_number(self):
        return self.page_number

    def set_bbox(self, bbox):
        self.bbox = bbox

    def get_bbox(self):
        return self.bbox