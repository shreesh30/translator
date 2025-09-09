from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Element:
    type: Optional[str] = None                  # e.g., "paragraph", "table", etc.
    font_size: float = field(default_factory=float)

    def get_type(self):
        return self.type

    def get_font_size(self):
        pass

    def set_volume(self, volume_name):
        pass

    def set_chapter(self, chapter_name):
        pass

    def set_font_size(self, font_size):
        pass