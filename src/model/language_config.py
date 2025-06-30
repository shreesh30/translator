from dataclasses import dataclass


@dataclass
class LanguageConfig:
    target_language: str
    target_language_key: str
    target_font_path: str
    font_multiplier: float
    line_spacing_multiplier: float
    source_language: str = None
    source_language_key: str = None
    right_to_left: bool = False

    def get_target_language(self):
        return self.target_language

    def get_target_language_key(self):
        return self.target_language_key

    def get_target_font_path(self):
        return self.target_font_path

    def get_font_multiplier(self):
        return self.font_multiplier

    def get_line_spacing_multiplier(self):
        return self.line_spacing_multiplier

    def get_right_to_left(self):
        return self.right_to_left

    def get_source_language(self):
        if self.source_language is None:
            return 'English'
        return self.source_language

    def get_source_language_key(self):
        if self.source_language_key is None:
            return 'eng_Latn'
        return self.source_language_key