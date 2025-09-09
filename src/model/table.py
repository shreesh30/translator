from dataclasses import dataclass, field
from typing import List

from src.model.element import Element
from src.model.footer import Footer
from src.model.line import Line



@dataclass
class Table:
    title: Line = field(default_factory=Line)
    sub_title: Line = field(default_factory=Line)
    rows: List[Line] = field(default_factory=list)
    columns: List[List[Line]] = field(default_factory=list)
    chapter: str = field(default_factory=str)
    volume: str = field(default_factory=str)
    content_table: bool = field(default_factory=bool)
    page_number: Footer = field(default_factory=Footer)
    type: str = field(default_factory=str)
    font_size: float = field(default_factory=float)

    def __post_init__(self):
        from src.utils.utils import Utils
        self.type = Utils.TYPE_TABLE

    def set_title(self, title):
        self.title = title

    def set_sub_title(self, sub_title):
        self.sub_title = sub_title

    def set_page_number(self, page_number):
        self.page_number = page_number

    def add_column(self, column):
        self.columns.append(column)

    def set_columns(self, columns):
        self.columns = columns

    def set_volume(self, volume):
        self.volume = volume

    def set_chapter(self, chapter):
        self.chapter = chapter

    def set_is_content_table(self, is_content_table):
        self.content_table  = is_content_table

    def get_title(self):
        return self.title

    def get_sub_title(self):
        return self.sub_title

    def get_page_number(self):
        return self.page_number

    def get_columns(self):
        return self.columns

    def get_chapter(self):
        return self.chapter

    def get_volume(self):
        return self.volume

    def is_content_table(self):
        return self.content_table

    def set_font_size(self, font_size):
        self.font_size=font_size

    def get_font_size(self):
        return self.font_size