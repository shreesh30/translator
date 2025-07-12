import fitz
import re
import string

from src.model.span import Span
from src.model.header import Header
from src.model.footer import Footer
from src.model.paragraph import Paragraph
from src.model.line import Line

from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class Page:
    number: int = field(default_factory=int)
    spans: List[Span]  = field(default_factory=list, repr=True)
    lines: List[Line] = field(default_factory=list, repr = False)
    paragraphs: List[Paragraph] = field(default_factory=list)
    headers: List[Header] = field(default_factory=list)
    footers: List[Footer] = field(default_factory=list)
    drawings: List[Dict] = field(default_factory=list, repr=False)
    min_x: float = field(default_factory=float)
    max_x:float = field(default_factory=float)
    min_y:float = field(default_factory=float)
    max_y:float = field(default_factory=float)
    content_width:float = field(default_factory=float)
    content_height:float = field(default_factory=float)
    target_language: str = field(default_factory=str, repr=False)
    extracted_page_number: str = field(default_factory=str)

    def _get_footer_line_y(self):
        """Returns the Y position of a full-width horizontal line if detected."""
        for drawing in self.drawings:
            rect = fitz.Rect(drawing['rect'])
            if (
                    int(rect.x0) == int(self.min_x) and
                    int(rect.x1) == int(self.max_x) and
                    int(rect.y0) == int(rect.y1)
            ):
                return rect.y1
        return None

    def _build_paragraph(self, lines):
        paragraph_centered = False
        for line in lines:
            left_indent = int(line.get_bbox().x0) - int(self.get_min_x())
            right_indent = int(self.get_max_x()) - int(line.get_bbox().x1)
            if abs(left_indent - right_indent) > 1:
                paragraph_centered = False
            else:
                paragraph_centered = True


        x0 = lines[0].get_bbox().x0
        y0 = lines[0].get_bbox().y0
        if paragraph_centered:
            x1= lines[0].get_bbox().x1
        else:
            x1 = lines[-1].get_bbox().x1
        y1 = lines[-1].get_bbox().y1

        para_bbox = fitz.Rect(x0, y0, x1, y1)
        font_size = lines[0].get_font_size()
        page_number = self.get_page_number()

        paragraph = Paragraph(page_number)
        paragraph.set_lines(lines)
        paragraph.set_font_size(font_size)
        paragraph.set_para_bbox(para_bbox)
        paragraph.set_start(para_bbox.x0)
        paragraph.set_end(para_bbox.x1)
        return paragraph


    def add_span(self, span):
        self.spans.append(span)

    def add_paragraph(self, para):
        self.paragraphs.append(para)

    def add_drawings(self, drawing):
        self.drawings.extend(drawing)

    def set_min_x(self, x):
        self.min_x = x

    def set_max_x(self, x):
        self.max_x = x

    def set_min_y(self, y):
        self.min_y = y

    def set_max_y(self, y):
        self.max_y = y

    def set_content_dimensions(self, min_x, max_x, min_y, max_y):
        self.set_min_x(min_x)
        self.set_max_x(max_x)
        self.set_min_y(min_y)
        self.set_max_y(max_y)

    def set_extracted_page_number(self, extracted_page_number):
        self.extracted_page_number = extracted_page_number

    def get_extracted_page_number(self):
        return self.extracted_page_number

    def get_paragraphs(self):
        return self.paragraphs

    def get_headers(self):
        return self.headers

    def get_footer(self):
        return self.footers

    def get_content_height(self):
        return self.content_height

    def get_content_width(self):
        return self.content_width

    def get_min_x(self):
        return self.min_x

    def get_max_x(self):
        return self.max_x

    def get_min_y(self):
        return self.min_y

    def get_max_y(self):
        return self.max_y

    def get_page_number(self):
        return self.number

    def compute_content_dimensions(self):
        self.content_width = self.max_x - self.min_x
        self.content_height = self.max_y - self.min_y

    def group_by_lines(self):
        """Group text spans into Line objects based on shared Y position.
        Adds space only when font size increases significantly within the same line.
        Stores the result in self.lines.
        """
        lines = []
        current_line = None

        for span in self.spans:
            if not span.get_text():
                continue

            is_new_line = (
                    current_line is None or
                    span.get_bbox().y1 - current_line.get_bbox().y1 > 1
            )

            if is_new_line:
                if current_line:
                    lines.append(current_line)

                current_line = Line(page_number=self.get_page_number())
                current_line.set_text(span.get_text().lower())
                current_line.set_bbox(fitz.Rect(span.get_bbox()))
                current_line.set_origin(span.get_origin())
                current_line.set_font_size(span.get_font_size())
                current_line.set_line_bbox(fitz.Rect(span.get_bbox()))
            else:
                # Update text
                current_line.set_text(current_line.get_text() + " " + span.get_text().lower())

                # Update bbox
                curr_bbox = span.get_bbox()
                updated_bbox = fitz.Rect(
                    current_line.get_bbox().x0,
                    current_line.get_bbox().y0,
                     curr_bbox.x1,
                     curr_bbox.y1,
                )
                current_line.set_bbox(updated_bbox)
                current_line.set_line_bbox(updated_bbox)

                # Update origin
                origin_x = min(current_line.get_origin()[0], span.get_origin()[0])
                origin_y = max(current_line.get_origin()[1], span.get_origin()[1])
                current_line.set_origin((origin_x, origin_y))

                # Update font size
                current_line.set_font_size(max(current_line.font_size, span.get_font_size()))


        if current_line:
            lines.append(current_line)

        self.lines = lines

    def group_by_paragraphs(self):
        """Group text lines into paragraphs based on vertical proximity.
        Stores the result in self.paragraphs.
        """
        paragraphs = []
        current_lines = []

        for i, line in enumerate(self.lines):
            is_new_para = (
                    i > 0 and (
                    line.get_page_number() != self.lines[i - 1].get_page_number() or
                    int(line.get_line_bbox().y0) - int(self.lines[i-1].get_line_bbox().y1)>=0
            )
            )

            if is_new_para and current_lines:
                paragraph_obj = self._build_paragraph(current_lines)
                paragraphs.append(paragraph_obj)
                current_lines = []

            current_lines.append(line)

        if current_lines:
            paragraph_obj = self._build_paragraph(current_lines)
            paragraphs.append(paragraph_obj)

        self.paragraphs = paragraphs

    def separate_content(self):
        footer_start = self._get_footer_line_y()

        def is_footer_by_line(line_obj):
            y0 = line_obj.get_line_bbox().y0
            return footer_start is not None and y0 > footer_start

        def is_footer_by_position(line_obj):
            y1 = line_obj.get_line_bbox().y1
            return (
                    int(y1) == int(self.max_y) and
                    line_obj.get_text().isdigit()
            )

        def is_header_by_position(line_obj):
            y1 = line_obj.get_line_bbox().y1
            return y1 < 131

        new_lines = []
        footers = []
        headers = []

        for line in self.lines:
            if is_footer_by_line(line) or is_footer_by_position(line):
                footers.append(line)
            elif is_header_by_position(line):
                headers.append(line)
            else:
                new_lines.append(line)

        return new_lines, headers, footers

    def process_footer(self, footers):
        new_footers = []

        if not footers:
            return

        footer = None

        for line in footers:
            line_text = line.get_text()

            if self.has_non_english_prefix(line_text):
                pattern = r'(?i)\benglish(?=\s+translation\s+of)'
                line_text = re.sub(pattern, self.target_language, line_text)

                footer = Footer()
                footer.set_text(line_text)
                footer.set_font_size(line.get_font_size())
                new_footers.append(footer)
            elif line_text.isdigit():
                footer = Footer()
                footer.set_text(line_text)
                footer.set_font_size(line.get_font_size())
                new_footers.append(footer)
                footer = None
            else:
                if footer:
                    # Append to the last footer's text
                    footer.set_text(footer.get_text() + " " + line_text)

        self.footers = new_footers

    def process_header(self, headers):
        if not headers:
            return

        new_headers = []

        for line in headers:
            line_text = line.get_text().strip()

            header = Header()
            header.set_text(line_text)
            new_headers.append(header)

        self.headers = new_headers

    # SEGREGATE HEADER, FOOTER AND MAIN CONTENT
    def process_lines(self):
        new_lines, headers, footers = self.separate_content()

        self.lines = new_lines
        self.process_header(headers)
        self.process_footer(footers)

    def normalize_spans(self):
        normalized = []
        prev_span = None
        punctuation_set = set(string.punctuation + '’‘“”—–')  # common typographic symbols

        for span in self.spans:
            if not span.get_text():
                continue

            text = span.get_text().strip()
            is_punctuation = all(char in punctuation_set for char in text)

            is_new_line = (
                    not prev_span or
                    abs(span.get_bbox().y1 - prev_span.get_bbox().y1) > 1.0
            )

            if is_new_line:
                if prev_span:
                    normalized.append(prev_span)
                prev_span = span
                continue

            if is_punctuation:
                prev_span.set_text("{0}{1}".format(prev_span.get_text(), span.get_text()))
                prev_span.get_bbox().x1 = span.get_bbox().x1
                continue

            x_gap = span.get_bbox().x0 - prev_span.get_bbox().x1

            if span.get_font_size() < prev_span.get_font_size():
                if x_gap <= 1.0:
                    prev_span.set_text("{0}{1}".format(prev_span.get_text(), span.get_text()))
                else:
                    prev_span.set_text("{0} {1}".format(prev_span.get_text(), span.get_text()))
                prev_span.get_bbox().x1 = span.get_bbox().x1
            else:
                normalized.append(prev_span)
                prev_span = span

        if prev_span:
            normalized.append(prev_span)

        self.spans = normalized

    def fix_punctuation_spacing(self):
        for line in self.lines:
            text = line.get_text()

            # Match abbreviations like B.N., c.i.e., H. V. R. (with optional space between)
            abbreviation_pattern = r'\b(?:[A-Za-z]\. ?){2,}'
            matches = re.findall(abbreviation_pattern, text)

            placeholder_map = {}

            for i, abbr in enumerate(matches):
                # Count how many actual dots (.) are in the abbreviation
                dot_count = abbr.count(".")

                # If it has more than 3 dots, insert a space after each dot
                if dot_count > 3:
                    # Remove existing spaces and re-insert one after every dot
                    cleaned = re.sub(r'\. ?', '.', abbr)  # Remove optional spaces
                    spaced = re.sub(r'\.', '. ', cleaned).strip()
                    abbr = spaced

                placeholder = f"<<ABBR_{i}>>"
                placeholder_map[placeholder] = abbr.upper()
                text = text.replace(matches[i], placeholder)

            # Fix general punctuation spacing
            text = re.sub(r'\s+([.,:;])', r'\1', text)  # "dec ." → "dec."
            text = re.sub(r'([.,:;])(?=\S)', r'\1 ', text)  # "dec.1946" → "dec. 1946"

            # Restore abbreviations safely
            for placeholder, abbr in placeholder_map.items():
                # Ensure a space exists after abbreviation if it's stuck to a word
                text = re.sub(rf'{re.escape(placeholder)}(?=\w)', abbr + ' ', text)
                # General replacement if not already handled
                text = text.replace(placeholder, abbr)

            # Final cleanup: remove extra spaces
            text = re.sub(r'\s{2,}', ' ', text)

            line.set_text(text)

    def map_footers_to_paragraphs(self):
        if not self.footers:
            return  # No footers to map

        for footer in self.footers:
            footer_text = footer.get_text()
            normalized_prefix = self.extract_prefix(footer_text).strip().lower()

            for paragraph in self.paragraphs:
                if self._footer_already_mapped(paragraph, footer_text):
                    continue

                for line in paragraph.get_lines():
                    normalized_line = line.get_text().strip().lower()
                    if normalized_prefix in normalized_line:
                        paragraph.add_footers(footer)
                        break  # Avoid mapping the same footer multiple times to one paragraph

    @staticmethod
    def _footer_already_mapped(paragraph, footer_text: str) -> bool:
        """Check if a footer is already mapped to a paragraph."""
        return footer_text in (f.get_text() for f in paragraph.get_footer())


    def extract_page_number(self):
        for footer in self.footers:
            footer_text = footer.get_text()
            if footer_text.isdigit():
                self.set_extracted_page_number(footer_text)

    @staticmethod
    def extract_prefix(text):
        match = re.match(r'^[^A-Za-z]+', text)
        return match.group(0) if match else ''

    def has_non_english_prefix(self,text):
        prefix = self.extract_prefix(text)
        return bool(re.search(r'[^A-Za-z\s]', prefix))


    def process_page(self):
        self.normalize_spans()
        self.group_by_lines()
        self.fix_punctuation_spacing()
        self.process_lines()
        self.group_by_paragraphs()
        self.compute_content_dimensions()
        self.map_footers_to_paragraphs()
        self.extract_page_number()