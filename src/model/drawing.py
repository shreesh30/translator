from dataclasses import field, dataclass

import fitz

from src.model.bbox import Bbox


@dataclass
class Drawing:
    page_number: int = field(default_factory=int)
    bbox:Bbox = field(default_factory=lambda: Bbox(0, 0, 0, 0))

    def set_page_number(self, page_number):
        self.page_number = page_number

    def get_page_number(self):
        return self.page_number

    def set_bbox(self, bbox):
        self.bbox = bbox

    def get_bbox(self):
        return self.bbox

