from dataclasses import dataclass, field

@dataclass
class Footer:
    text: str= field(default_factory=str)

    def set_text(self, text):
        self.text = text

    def get_text(self):
        return self.text