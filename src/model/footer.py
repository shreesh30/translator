from dataclasses import dataclass, field


@dataclass
class Footer:
    text: str = field(default_factory=str)
    font_size: float = field(default_factory=float)

    def set_text(self, text):
        self.text = text

    def get_text(self) -> str:
        return self.text

    def set_font_size(self, font_size):
        self.font_size = font_size

    def get_font_size(self):
        return self.font_size
