import re
from dataclasses import dataclass, field
from typing import List
import logging

import fitz

from src.model.drawing import Drawing
from src.model.element import Element
from src.model.footer import Footer
from src.model.header import Header
from src.model.line import Line, TableLine
from src.model.paragraph import Paragraph
from src.model.table import Table

logger = logging.getLogger(__name__)
@dataclass
class Page:
    number: int = field(default_factory=int)
    lines: List[Line] = field(default_factory=list, repr = False)
    elements: List[Element] = field(default_factory=list)
    paragraphs: List[Paragraph] = field(default_factory=list, repr=False)
    tables: List[Table]  = field(default_factory=list, repr=False)
    headers: List[Header] = field(default_factory=list)
    footers: List[Footer] = field(default_factory=list)
    drawings: List[Drawing] = field(default_factory=list)
    min_x: float = field(default_factory=float)
    max_x:float = field(default_factory=float)
    min_y:float = field(default_factory=float)
    max_y:float = field(default_factory=float)
    content_width:float = field(default_factory=float)
    content_height:float = field(default_factory=float)
    # target_language: str = field(default_factory=str, repr=False)
    extracted_page_number: str = field(default_factory=str)
    line_spacing: int = field(default_factory=int)
    is_content_table: bool = field(default_factory=bool)

    def _get_footer_line_y(self):
        """Returns the Y position of a full-width horizontal line if detected."""
        for drawing in self.drawings:
            bbox = drawing.get_bbox()
            if (
                    int(bbox.x0) == int(self.min_x) and
                    int(bbox.x1) == int(self.max_x) and
                    int(bbox.y0) == int(bbox.y1)
            ):
                return bbox.y1
        return None

    def _build_paragraph(self, lines):
        paragraph_centered = False
        for line in lines:
            left_indent = int(line.get_line_bbox().x0) - int(self.get_min_x())
            right_indent = int(self.get_max_x()) - int(line.get_line_bbox().x1)
            if abs(left_indent - right_indent) > 1:
                paragraph_centered = False
            else:
                paragraph_centered = True


        x0 = lines[0].get_line_bbox().x0
        y0 = lines[0].get_line_bbox().y0
        if paragraph_centered:
            x1= lines[0].get_line_bbox().x1
        else:
            x1 = lines[-1].get_line_bbox().x1
        y1 = lines[-1].get_line_bbox().y1

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

    def add_line(self, line):
        self.lines.append(line)

    def add_element(self, element):
        self.elements.append(element)

    def add_elements(self, elements):
        self.elements.extend(elements)

    def add_paragraph(self, para):
        self.paragraphs.append(para)

    def add_table(self, table):
        self.tables.append(table)

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

    def set_line_spacing(self, line_spacing):
        self.line_spacing = line_spacing

    def set_is_content_table(self, is_content_table):
        self.is_content_table = is_content_table

    def get_is_content_table(self):
        return self.is_content_table

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

    def get_drawings(self):
        return self.drawings

    def get_line_spacing(self):
        return self.line_spacing

    def get_tables(self):
        return self.tables

    def get_elements(self):
        return self.elements

    def compute_content_dimensions(self):
        self.content_width = self.max_x - self.min_x
        self.content_height = self.max_y - self.min_y

    def group_lines(self):
        lines = []
        current_line = None

        for line in self.lines:
            if not line.get_text() or isinstance(line, TableLine):
                continue

            is_new_line = (current_line is None or abs(line.get_line_bbox().y1 - current_line.get_line_bbox().y1) >1)

            if is_new_line:
                if current_line:
                    lines.append(current_line)

                current_line = line
            else:
                current_line.set_text(current_line.get_text()+ " "+ line.get_text() )

                current_bbox = line.get_line_bbox()

                updated_bbox = fitz.Rect(
                    current_line.get_line_bbox().x0,
                    current_line.get_line_bbox().y0,
                    current_bbox.x1,
                    current_bbox.y1
                )
                current_line.set_line_bbox(updated_bbox)
                current_line.set_font_size(max(current_line.font_size, line.get_font_size()))

        if current_line:
            lines.append(current_line)

        self.lines = lines

        return lines

    def group_by_paragraphs(self, lines):
        """Group text lines into paragraphs based on vertical proximity.
        Stores the result in self.paragraphs.
        """
        paragraphs = []
        current_lines = []

        for i, line in enumerate(lines):
            is_new_para = (
                    i > 0 and (
                    line.get_page_number() != lines[i - 1].get_page_number() or
                    line.get_line_bbox().y0 - lines[i-1].get_line_bbox().y1>self.line_spacing
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

        self.add_elements(paragraphs)
        # self.elements.extend(paragraphs)

    def separate_content(self, lines):
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

        for line in lines:
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
                # pattern = r'(?i)\benglish(?=\s+translation\s+of)'
                # line_text = re.sub(pattern, self.target_language, line_text)

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

    @staticmethod
    def find_line_spacing(lines):
        line_spacing = 0

        if not lines:
            return line_spacing

        for i,line in enumerate(lines):
              if i>0:
                  line_spacing += line.get_line_bbox().y0-lines[i-1].get_line_bbox().y1

        line_spacing = line_spacing/len(lines)

        return line_spacing

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
    def process_lines(self, lines):
        new_lines, headers, footers = self.separate_content(lines)

        self.process_header(headers)
        self.process_footer(footers)

        return new_lines

    def fix_punctuation_spacing(self, lines):
        for line in lines:
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

            # # Fix general punctuation spacing
            # text = re.sub(r'\s+([.,:;])', r'\1', text)  # "dec ." → "dec."
            # text = re.sub(r'([.,:;])(?=\S)', r'\1 ', text)  # "dec.1946" → "dec. 1946"

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

            if footer_text.isdigit():
                continue

            for element in self.get_elements():
                if isinstance(element, Table) or  (isinstance(element, Paragraph) and self._footer_already_mapped(element, footer_text)):
                    continue

                if isinstance(element, Paragraph):
                    for line in element.get_lines():
                        normalized_line = line.get_text().strip().lower()
                        if normalized_prefix in normalized_line:
                            element.add_footers(footer)
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
        match = re.match(r'^[^\w\s]+', text)  # match non-alphanumeric, non-space characters
        return match.group(0) if match else ''

    def has_non_english_prefix(self,text):
        prefix = self.extract_prefix(text)
        return bool(re.search(r'[^A-Za-z\s]', prefix))

    @staticmethod
    def is_page_number(text):
        # Strip leading/trailing whitespace
        text = text.strip()

        # Match single page numbers: e.g., "1", "55"
        if re.fullmatch(r"\d{1,3}", text):
            return True

        # Match page ranges: "1-2", "1—2", "55—56", etc.
        if re.fullmatch(r"\d{1,3}[\-—–]\d{1,3}", text):
            return True

        return False

    def merge_lines(self, lines, line_spacing):

        # line_spacing = self.avg_line_spacing(lines)
        logger.info(f'Line Spacing: {line_spacing}')

        new_lines = []

        current_lines = []

        for i, line in enumerate(lines):
            is_new_line = (i>0 and (line.get_line_bbox().y0-lines[i-1].get_line_bbox().y1)>line_spacing)

            if is_new_line and current_lines:
                line_obj =  self._merge_lines(current_lines)
                new_lines.append(line_obj)
                current_lines = []

            current_lines.append(line)

        if current_lines:
            line_obj = self._merge_lines(current_lines)
            new_lines.append(line_obj)

        return new_lines

    @staticmethod
    def _merge_lines(lines):
        line_text = ""
        for i, line in enumerate(lines):
            text = line.get_text().strip()

            if text.endswith("-"):
                # Remove the hyphen and don't add space
                line_text += text[:-1]
            else:
                # Add text and a space after it (unless it's the last line)
                line_text += text
                if i != len(lines) - 1:
                    line_text += " "

        font_size = lines[-1].get_font_size()

        x0 = lines[0].get_line_bbox().x0
        y0 = lines[0].get_line_bbox().y0
        x1 = lines[-1].get_line_bbox().x1
        y1 = lines[-1].get_line_bbox().y1
        line_bbox = fitz.Rect(x0, y0, x1, y1)

        line = Line(text=line_text, font_size=font_size, line_bbox=line_bbox)

        return line

    def process_contents_table(self, center, left_column, right_column):
        filtered_left_column = [line for line in left_column if not self.is_page_number(line.text)]
        filtered_right_column = [line for line in right_column if not self.is_page_number(line.text)]

        center = self.merge_lines(center, 0)
        left_column = self.merge_lines(filtered_left_column, self.find_line_spacing(filtered_left_column))
        right_column = self.merge_lines(filtered_right_column, self.find_line_spacing(filtered_right_column))

        table =  Table()
        for line in center:
            text = line.get_text()

            if self.is_roman_numeral(text):
                footer = Footer(text, line.get_font_size())
                table.set_page_number(footer)
                continue

            if not table.get_title().get_text():
                table.set_title(line)
            else:
                table.set_sub_title(line)

        table.add_column(left_column)
        table.add_column(right_column)
        table.set_is_content_table(True)

        logger.info(f'Table of Content: {table}')

        return table

    @staticmethod
    def is_roman_numeral(text):
        """
        Check if text is a Roman numeral (including wrapped in brackets/parentheses).
        """
        cleaned = re.sub(r'[^a-zA-Z]', '', text.strip().upper())

        if not cleaned:  # guard against empty strings
            return False

        pattern = r'^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$'
        return bool(re.fullmatch(pattern, cleaned))

    def extract_contents_table(self):
        all_x_0 = [line.get_line_bbox().x0 for line in self.lines if isinstance(line, TableLine)]
        all_x_1 = [line.get_line_bbox().x1 for line in self.lines if isinstance(line, TableLine)]

        min_x = min(all_x_0)
        max_x = max(all_x_1)
        mid_x = (min_x + max_x) / 2

        center = []
        left_column = []
        right_column = []

        for line in self.lines:
            if isinstance(line,TableLine):
                line_bbox = line.get_line_bbox()
                left_indent = int(line_bbox.x0) - int(min_x)
                right_indent = int(max_x) - int(line_bbox.x1)
                is_centered = abs(left_indent - right_indent) <= 1

                if is_centered and left_indent != 0 and right_indent != 0:
                    center.append(line)
                elif line_bbox.x0 < mid_x:
                    left_column.append(line)
                elif line_bbox.x0 >= mid_x:
                    right_column.append(line)

        if left_column and right_column:
            left_column = sorted(left_column, key=lambda l: l.get_line_bbox().y0)
            right_column = sorted(right_column, key=lambda l: l.get_line_bbox().y0)

            logger.info(f'Center: {center}')
            logger.info(f'Left Column: {left_column}')
            logger.info(f'Right Column: {right_column}')

            table = self.process_contents_table(center, left_column, right_column)
            self.add_element(table)


    def process_page(self):
        if self.is_content_table:
            self.extract_contents_table()
        else:
            lines = self.group_lines()
            self.fix_punctuation_spacing(lines)
            lines = self.process_lines(lines)
            self.set_line_spacing(self.find_line_spacing(lines))
            self.group_by_paragraphs(lines)
            self.compute_content_dimensions()
            self.map_footers_to_paragraphs()
            self.extract_page_number()