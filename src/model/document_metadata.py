from dataclasses import dataclass, field
from typing import List

from src.model.page import Page


@dataclass
class DocumentMetadata:
    paragraph_start: int =  field(default_factory=int, repr=False)
    pages: List[Page] = field(default_factory=list)

    def get_pages(self):
        return self.pages

    def get_paragraph_start(self):
        return self.paragraph_start

    def get_page_number_info(self):
        extracted_page_number = None
        page_number_start = None

        for page in self.pages:
            if page.get_extracted_page_number() and extracted_page_number is None:
                extracted_page_number = int(page.get_extracted_page_number())
                page_number_start = page.get_page_number()
                break

        return extracted_page_number, page_number_start