from dataclasses import dataclass, field


@dataclass
class LanguageConfig:
    target_language: str = field(default_factory=str)
    target_language_key: str = field(default_factory=str)
    target_font_path: str = field(default_factory=str)
    font_size_multiplier: float = field(default_factory=float)
    line_spacing_multiplier: float = field(default_factory=float)
    target_font_name: str = field(default_factory=str)
    source_language: str = field(default_factory=str)
    source_language_key: str = field(default_factory=str)
    right_to_left: bool = field(default_factory=bool)

    def get_target_language(self):
        return self.target_language

    def get_target_language_key(self):
        return self.target_language_key

    def get_target_font_path(self):
        return self.target_font_path

    def get_font_size_multiplier(self):
        return self.font_size_multiplier

    def get_line_spacing_multiplier(self):
        return self.line_spacing_multiplier

    def get_right_to_left(self):
        return self.right_to_left

    def get_source_language(self):
        if not self.source_language:
            return 'English'
        return self.source_language

    def get_source_language_key(self):
        if not self.source_language_key:
            return 'eng_Latn'
        return self.source_language_key

    def get_target_font_name(self):
        return self.target_font_name

    def set_target_font_name(self, font_name):
        self.target_font_name = font_name
