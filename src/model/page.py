import fitz
import re

from src.model.header import Header
from src.model.footer import Footer
from src.model.paragraph import Paragraph
from src.model.line import Line

from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class Page:
    number: int = field(default_factory=int)
    spans: List[Dict]  = field(default_factory=list, repr=False)
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

    @staticmethod
    def _build_paragraph(lines):
        x0s = [line.get_bbox().x0 for line in lines]
        y0s = [line.get_bbox().y0 for line in lines]
        x1s = [line.get_bbox().x1 for line in lines]
        y1s = [line.get_bbox().y1 for line in lines]

        para_bbox = fitz.Rect(min(x0s), min(y0s), max(x1s), max(y1s))
        font_size = lines[0].get_font_size()
        page_number = lines[0].get_page_number()

        paragraph = Paragraph(page_number)
        paragraph.set_lines(lines)
        paragraph.set_font_size(font_size)
        paragraph.set_para_bbox(para_bbox)

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

    def sort_spans(self):
        self.spans = sorted(
            self.spans,
            key=lambda x: (x["origin"][1])
        )

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
        prev_span = None

        def should_insert_space(previous_span, curr_span):
            if not previous_span:
                return False

            x_gap = curr_span["bbox"].x0 - previous_span["bbox"].x1
            return x_gap > 1.0

        for span in self.spans:
            if not span.get("text"):
                continue

            is_new_line = (
                    current_line is None or
                    int(span["bbox"].y1) != int(current_line.get_bbox().y1)
            )

            if is_new_line:
                if current_line:
                    lines.append(current_line)
                current_line = Line(page_number=self.number)
                current_line.set_text(span["text"])
                current_line.set_line_bbox(fitz.Rect(span["line_bbox"]))
                current_line.set_bbox(fitz.Rect(span["bbox"]))
                current_line.set_origin(span["origin"])
                current_line.set_font_size(span["size"])
            else:
                # Update text
                if should_insert_space(prev_span, span):
                    current_line.set_text(current_line.get_text() + " " + span["text"])
                else:
                    current_line.set_text(current_line.get_text() + span["text"])

                # Update bbox
                curr_bbox = span["bbox"]
                updated_bbox = fitz.Rect(
                    min(current_line.get_bbox().x0, curr_bbox.x0),
                    min(current_line.get_bbox().y0, curr_bbox.y0),
                    max(current_line.get_bbox().x1, curr_bbox.x1),
                    max(current_line.get_bbox().y1, curr_bbox.y1),
                )
                current_line.set_bbox(updated_bbox)

                # Update origin
                origin_x = min(current_line.get_origin()[0], span["origin"][0])
                origin_y = max(current_line.get_origin()[1], span["origin"][1])
                current_line.set_origin((origin_x, origin_y))

                # Update font size
                current_line.set_font_size(max(current_line.font_size, span["size"]))

            prev_span = span

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
                    (int(line.get_line_bbox().y0) - int(self.lines[i-1].get_line_bbox().y1)>2)
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

            if '*' in line_text:
                footer = Footer()
                footer.set_text(line_text)
                new_footers.append(footer)
            elif line_text.isdigit():
                footer = Footer()
                footer.set_text(line_text)
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
        prev_span = None
        current_line_font_size = None

        for span in self.spans:
            if not span.get("text"):
                continue

            # Check if we're starting a new line
            is_new_line = (
                    prev_span is None or
                    int(span["bbox"].y1) != int(prev_span["bbox"].y1)
            )

            if is_new_line:
                current_line_font_size = span["size"]  # Set new line's base font size
            else:
                if span["size"] != current_line_font_size:
                    old_size = span["size"]
                    scaling_factor = current_line_font_size / old_size

                    # Update font size
                    span["size"] = current_line_font_size

                    # Scale bbox height
                    bbox = span["bbox"]
                    height = bbox.y1 - bbox.y0
                    new_height = height * scaling_factor
                    baseline = bbox.y1  # assume bottom aligned
                    new_y0 = baseline - new_height
                    span["bbox"] = fitz.Rect(bbox.x0, new_y0, bbox.x1, baseline)

            prev_span = span

    def process_page(self):
        self.sort_spans()
        self.normalize_spans()
        self.group_by_lines()
        self.process_lines()
        self.group_by_paragraphs()
        self.compute_content_dimensions()



