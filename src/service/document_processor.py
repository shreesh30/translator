import re
from collections import Counter, defaultdict
from typing import List, Tuple
import logging
import fitz

from src.model.drawing import Drawing
from src.model.page import Page
from src.model.paragraph import Paragraph
from src.model.span import Span
from src.model.line import Line, TableLine
from src.model.table import Table

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, input_folder_path):
        self.input_folder_path = input_folder_path
        self.pages = None
        self.paragraphs = None
        self.tables = None
        self.elements = None

    def set_paragraphs(self, paragraphs):
        self.paragraphs = paragraphs

    def add_paragraphs(self, paragraphs):
        self.paragraphs.extend(paragraphs)

    def get_paragraphs(self):
        return self.paragraphs

    def get_elements(self):
        return self.elements

    def set_pages(self, pages):
        self.pages = pages

    def add_pages(self, pages):
        self.pages.extend(pages)

    def add_page(self, page):
        self.pages.append(page)

    def get_pages(self):
        return self.pages

    def merge_paragraphs(self):
        merged_paragraphs = []
        prev_paragraph = None

        index = 0
        while index < len(self.elements):
            element = self.elements[index]

            if not isinstance(element, Paragraph):
                break

            paragraph = element
            curr_page_num = paragraph.get_page_number()
            curr_bbox = paragraph.get_para_bbox()
            curr_start_x = int(curr_bbox.x0) if curr_bbox else None
            page_min_x = int(self.pages[curr_page_num].get_min_x())

            is_merge_case = (
                    prev_paragraph is not None and
                    prev_paragraph.get_page_number() != curr_page_num and
                    curr_start_x == page_min_x
            )

            if is_merge_case:
                # Merge prev and current paragraph
                new_para = Paragraph()
                new_para.set_lines(prev_paragraph.get_lines() + paragraph.get_lines())
                new_para.set_font_size(prev_paragraph.get_font_size())
                new_para.set_para_bbox(fitz.Rect(prev_paragraph.get_para_bbox().x0, prev_paragraph.get_para_bbox().y0,
                                                 paragraph.get_para_bbox().x1, paragraph.get_para_bbox().y1))
                new_para.set_start(prev_paragraph.get_start())
                new_para.set_end(paragraph.get_end())
                new_para.set_page_number(prev_paragraph.get_page_number())

                combined_footers = prev_paragraph.get_footer() + paragraph.get_footer()
                for footer in combined_footers:
                    new_para.add_footers(footer)

                merged_paragraphs.append(new_para)
                prev_paragraph = None
            else:
                if prev_paragraph:
                    merged_paragraphs.append(prev_paragraph)
                prev_paragraph = paragraph

            index += 1

        if prev_paragraph:
            merged_paragraphs.append(prev_paragraph)

        # Replace only the paragraph portion in self.elements
        self.elements = merged_paragraphs + self.elements[index:]

    def get_paragraph_start(self):
        paragraphs = []

        for element in self.elements:
            if isinstance(element, Paragraph):
                paragraphs.append(element)

        if not paragraphs:
            return 0

        para_start_values = [para.get_start() for para in paragraphs]
        counter = Counter(para_start_values)

        if not counter:
            return 0  # Again, default fallback

        para_start = counter.most_common(1)[0][0]
        return para_start

    def get_avg_font_size(self):
        font_size = 0
        paragraphs = []
        for element in self.elements:
            if isinstance(element, Paragraph):
                paragraphs.append(element)

        for para in paragraphs:
            font_size+=para.get_font_size()


        return int(font_size/len(paragraphs))

    def create_sub_paragraphs(self):
        para_start = self.get_paragraph_start()
        avg_font_size = self.get_avg_font_size()

        for element in self.elements:

            if isinstance(element, Paragraph):
                paragraph = element
                original_lines = paragraph.get_lines()
                new_main_lines = []
                i = 0

                while i < len(original_lines):
                    line = original_lines[i]
                    page = self.pages[line.get_page_number()]
                    min_x = page.get_min_x()
                    max_x = page.get_max_x()
                    line_bbox = line.get_line_bbox()

                    is_sub_para = False

                    if i > 0 and line.get_font_size()>=avg_font_size:
                        prev_line = original_lines[i-1]
                        # Left-aligned logic
                        if int(line_bbox.x0) == int(para_start) or (int(original_lines[i-1].get_line_bbox().x1)!=int(max_x) and not self.is_centered(prev_line,prev_line.get_page_number())):
                            is_sub_para = True

                        #  Right-aligned logic
                        elif int(line_bbox.x0) != int(para_start) and int(line_bbox.x0) != int(min_x) and int(
                                line_bbox.x1) == int(max_x):
                            is_sub_para = True

                            #  Center-aligned logic
                        elif self.is_centered(line, line.get_page_number()):
                            is_sub_para = False

                    new_sub_para = is_sub_para

                    if i == 0:
                        # Always keep the first line in the main paragraph
                        new_main_lines.append(line)
                        i += 1
                        continue

                    if new_sub_para:
                        # Start new sub-paragraph
                        sub_paragraph = Paragraph(
                            page_number=line.get_page_number(),
                            lines=[line],
                            font_size=line.get_font_size(),
                            para_bbox=line_bbox,
                            start=line_bbox.x0,
                            end=line_bbox.x1
                        )
                        paragraph.add_sub_paragraph(sub_paragraph)
                        # Remove this line from main paragraph
                        i += 1

                        # Add following non-indented lines to this sub-paragraph
                        while i < len(original_lines):
                            next_line = original_lines[i]
                            next_bbox = next_line.get_line_bbox()
                            next_min_x = self.pages[next_line.get_page_number()].get_min_x()

                            next_is_indented = int(next_bbox.x0) != int(next_min_x)

                            if next_is_indented:
                                # Start of a new sub-paragraph → break
                                break

                            # Add to current sub-paragraph
                            sub_paragraph.get_lines().append(next_line)
                            bbox = sub_paragraph.get_para_bbox()
                            bbox.y1 = max(bbox.y1, next_bbox.y1)
                            bbox.x1 = max(bbox.x1, next_bbox.x1)
                            sub_paragraph.set_end(bbox.x1)

                            i += 1
                    else:
                        new_main_lines.append(line)
                        i += 1

                # Set only top-level lines to the paragraph
                paragraph.set_lines(new_main_lines)

    def get_page_number_info(self):
        extracted_page_number = 0
        page_number_start = 0

        for page in self.pages:
            if page.get_extracted_page_number() and extracted_page_number==0:
                extracted_page_number = int(page.get_extracted_page_number())
                page_number_start = page.get_page_number()
                break

        return extracted_page_number, page_number_start

    @staticmethod
    def _is_footer_drawing(drawing, page):
        """Determines if a drawing is a full-width footer line."""
        bbox = drawing.get_bbox()
        left_indent = int(bbox.x0) - int(page.get_min_x())
        right_indent = int(page.get_max_x()) - int(bbox.x1)
        return left_indent == 0 and right_indent == 0

    def is_centered(self, element, page_number):
        page = self.pages[page_number]
        bbox = fitz.Rect()

        if isinstance(element, Paragraph):
            bbox = element.get_para_bbox()
        elif isinstance(element, Line):
            bbox = element.get_line_bbox()

        left_indent = int(bbox.x0) - int(page.get_min_x())
        right_indent = int(page.get_max_x()) - int(bbox.x1)

        return abs(left_indent - right_indent) <= 1 and left_indent != 0 and right_indent != 0

    def _get_non_footer_drawings(self):
        """Returns a list of non-footer drawings from all pages."""
        non_footer_drawings = []

        for page in self.pages:
            for drawing in page.drawings:
                if not self._is_footer_drawing(drawing, page):
                    non_footer_drawings.append(drawing)

        return non_footer_drawings

    @staticmethod
    def extract_volume_name(headers):
        for header in headers:
            # Remove leading page number (e.g., '686 ')
            cleaned = re.sub(r'^\d+\s*', '', header).strip().lower()
            # Check if the cleaned header starts with the known title
            if cleaned.startswith("constituent assembly of india"):
                return cleaned
        return None

    def add_volume(self):
        headers = [header.get_text() for page in self.pages for header in page.get_headers()]
        volume_name = self.extract_volume_name(headers)

        for element in self.elements:
            element.set_volume(volume_name)

    def add_chapters(self):
        drawings = self._get_non_footer_drawings()
        page_to_elements = self._group_elements_by_page()
        chapter_names = self._extract_chapter_names(drawings, page_to_elements)

        if chapter_names:
            self._assign_chapters_to_elements(chapter_names)

    def _group_elements_by_page(self):
        """Groups paragraphs by their page number."""
        page_dict = defaultdict(list)
        # for paragraph in self.paragraphs:
        #     page_number = paragraph.get_page_number()
        #     page_dict[page_number].append(paragraph)

        for element in self.elements:
            page_number = element.get_page_number()
            if isinstance(page_number, int):
                page_dict[page_number].append(element)

        return page_dict

    def _extract_chapter_names(self, drawings, page_dict):
        """Extracts chapter names that appear immediately after drawing elements."""
        chapters = []
        for drawing in drawings:
            page_number = drawing.get_page_number()
            bbox = drawing.get_bbox()
            elements = page_dict.get(page_number, [])

            for element in elements:
                if isinstance(element, Table):
                    continue

                paragraph = element
                if paragraph.get_para_bbox().y0 >= bbox.y1:
                    if self.is_centered(paragraph, page_number):
                        chapter_text = " ".join(line.get_text() for line in paragraph.get_lines())
                        chapters.append(chapter_text)
                    break  # Only check the first paragraph below the drawing
        return chapters

    def _assign_chapters_to_elements(self, chapter_names):
        """Assigns the appropriate chapter name to each paragraph and its sub-paragraphs."""
        current_chapter = None

        for element in self.elements:
            if isinstance(element, Paragraph):
                paragraph = element
                para_text = " ".join(line.get_text() for line in paragraph.get_lines())
                if para_text in chapter_names:
                    current_chapter = para_text.strip()

            if current_chapter:
                element.set_chapter(current_chapter)
                if isinstance(element, Paragraph):
                    paragraph = element
                    for sub_para in paragraph.get_sub_paragraphs():
                        sub_para.set_chapter(current_chapter)

    @staticmethod
    def _parse_span(span: dict, page_num: int):
        """
        Parses a span and returns cleaned span information.
        Converts ALL-UPPERCASE spans to Title Case (e.g., 'PREFACE' -> 'Preface', 'SIR B.N. RAU' -> 'Sir B.N. Rau').
        """
        text = span["text"]
        if not text:
            return {}

        font = span["font"]

        # Preserve bold styling if present in font name
        if "bold" in font.lower():
            text = f"[123]{text}[456]"

        new_span = Span()
        new_span.set_text(text)
        new_span.set_font(font)
        new_span.set_font_size(span["size"])
        new_span.set_page_num(page_num)
        new_span.set_origin(span["origin"])
        new_span.set_bbox(fitz.Rect(span["bbox"]))

        return new_span

    def extract_pages(self):
        doc = fitz.open(self.input_folder_path)
        self.pages = [self._process_page(page) for page in doc]

    def _process_page(self, page):
        page_num = page.number
        pg = Page(number=page_num)

        if self.has_center_vertical_line(page) and self.is_table_of_content(page):
            self._process_contents_table(pg, page)
        else:
            self._process_regular_page(pg, page)


        return pg

    @staticmethod
    def _is_single_two_column_table(tables):
        return len(tables.tables) == 1 and tables.tables[0].col_count == 2

    def _process_regular_page(self, pg, page):
        page_num = page.number
        blocks = page.get_text("dict", sort=True)["blocks"]

        # Process drawings
        for drawing in page.get_drawings():
            bbox = fitz.Rect(drawing.get("rect"))
            pg.add_drawings([Drawing(bbox=bbox, page_number=page_num)])

        # Set page dimensions
        min_x, max_x, min_y, max_y = self.get_content_dimensions(blocks)
        pg.set_content_dimensions(min_x, max_x, min_y, max_y)

        # Process text lines and spans
        for block in blocks:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                line_obj = Line(page_number=page_num)
                line_obj.set_line_bbox(fitz.Rect(line.get("bbox")))
                for span in line.get("spans", []):
                    new_span = self._parse_span(span, page_num)
                    text = line_obj.get_text().lower() + new_span.get_text().lower()
                    line_obj.set_text(text)
                    line_obj.set_font_size(new_span.get_font_size())
                pg.add_line(line_obj)

    def _process_contents_table(self, pg, page):
        blocks = page.get_text("dict", sort=True).get("blocks", [])

        for block in blocks:
            for line in block.get("lines",[]):
                line_text = "".join(span.get("text", "") for span in line.get("spans", []))
                line_bbox = fitz.Rect(line.get("bbox"))
                font_size = self.get_font_size(line)

                line = TableLine(line_bbox=line_bbox, text=line_text, font_size=font_size)

                pg.add_line(line)

        pg.set_is_content_table(True)

    @staticmethod
    def is_table_of_content(page):
        """
        Check if the given blocks represent a table of contents.
        Returns True if either:
        1. There's a centered block containing "Contents" (case-insensitive), or
        2. Any line contains "Pages" (case-insensitive)
        """
        blocks = page.get_text("dict", sort=True).get("blocks", [])
        # Calculate page boundaries
        all_x_0 = [b["bbox"][0] for b in blocks]
        all_x_1 = [b["bbox"][2] for b in blocks]
        min_x, max_x = min(all_x_0), max(all_x_1)

        logger.debug(f"min_x: {min_x}, max_x: {max_x}")

        for block in blocks:
            for line in block.get("lines", []):
                line_text = "".join(span.get("text", "") for span in line.get("spans", []))
                line_text_lower = line_text.lower()

                # Check for "Pages" keyword (anywhere in any line)
                if "pages" in line_text_lower:
                    return True

                # Check for centered "Contents" block
                line_bbox = fitz.Rect(line["bbox"])
                left_indent = int(line_bbox.x0) - int(min_x)
                right_indent = int(max_x) - int(line_bbox.x1)

                # Considered centered if indents differ by <= 1 unit and both indents exist
                is_centered = abs(left_indent - right_indent) <= 1
                has_indents = left_indent > 0 and right_indent > 0

                if is_centered and has_indents and "contents" in line_text_lower:
                    return True

        return False

    @staticmethod
    def has_center_vertical_line(page, tolerance=10):
        center_x = page.rect.width / 2
        drawings = page.get_drawings()

        for drawing in drawings:
            if drawing["type"] != "s":
                continue

            for item in drawing["items"]:
                if item[0] == "l":  # 'l' means line
                    x0, y0 = item[1]
                    x1, y1 = item[2]

                    # Check if it's a vertical line
                    if abs(x0 - x1) < 1e-2:
                        # Check if it's near the vertical center of the page
                        if abs(x0 - center_x) <= tolerance:
                            return True
        return False

    @staticmethod
    def get_font_size(line):
        """Safely get the font size from the last span in a line"""
        try:
            spans = line.get("spans", [])
            if spans:  # Check if spans exist
                last_span = spans[-1]
                return last_span.get("size", 0)
            return 0  # Return default if no spans
        except (AttributeError, IndexError, TypeError):
            return 0  # Return default if any error occurs

    @staticmethod
    def get_content_dimensions(blocks: List[dict]) -> Tuple[float, float, float, float]:
        """
        Calculates the effective content width by finding the min/max X coordinates
        of all text spans in text blocks.
        """
        min_x = float('inf')
        max_x = float('-inf')
        min_y = float('inf')
        max_y = float('-inf')

        for block in blocks:
            if block["type"] != 0:  # Only text blocks
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    x0, y0, x1, y1 = span["bbox"]
                    min_x = min(min_x, x0)
                    max_x = max(max_x, x1)
                    min_y = min(min_y, y0)
                    max_y = max(max_y, y1)

        if min_x == float('inf') or max_x == float('-inf') or min_y == float('inf') or max_y == float('-inf'):
            return 0, 0, 0, 0  # No text found

        return min_x, max_x, min_y, max_y


    def process_document(self):
        logger.info('Processing Document')
        try:
            self.extract_pages()

            for page in self.pages:
                logger.debug(
                    f"Page {page.get_page_number()} Dimensions:\n"
                    f"Min X: {page.get_min_x()}\n"
                    f"Max X: {page.get_max_x()}\n"
                    f"Min Y: {page.get_min_y()}\n"
                    f"Max Y: {page.get_max_y()}"
                )
                page.process_page()
                logger.debug(f'Page: {page}')

            self.elements = [element for page in self.pages for element in page.get_elements()]

            paragraph_exists = False
            for element in self.elements:
                if isinstance(element, Paragraph):
                    paragraph_exists = True
                    break

            if paragraph_exists:
                self.merge_paragraphs()
                self.create_sub_paragraphs()
                self.add_chapters()
                self.add_volume()
        except Exception as e:
            logger.error(f'Error while processing document: {e}')