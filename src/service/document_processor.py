import re
from collections import Counter, defaultdict
from typing import List

import fitz

from src.model.page import Page
from src.model.paragraph import Paragraph


class DocumentProcessor:
    def __init__(self, pages: List[Page]):
        self.pages = pages
        self.paragraphs = None

    def set_paragraphs(self, paragraphs):
        self.paragraphs = paragraphs

    def add_paragraphs(self, paragraphs):
        self.paragraphs.extend(paragraphs)

    def get_paragraphs(self):
        return self.paragraphs

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

        for paragraph in self.paragraphs:
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

                # Merge footers correctly
                combined_footers = prev_paragraph.get_footer() + paragraph.get_footer()
                for footer in combined_footers:
                    new_para.add_footers(footer)

                merged_paragraphs.append(new_para)
                prev_paragraph = None
            else:
                if prev_paragraph:
                    merged_paragraphs.append(prev_paragraph)
                prev_paragraph = paragraph

        if prev_paragraph:
            merged_paragraphs.append(prev_paragraph)

        self.paragraphs = merged_paragraphs

    def get_paragraph_start(self):
        if not self.paragraphs:
            return 0  # Or some appropriate default value

        para_start_values = [para.get_start() for para in self.paragraphs]

        counter = Counter(para_start_values)

        if not counter:
            return 0  # Again, default fallback

        para_start = counter.most_common(1)[0][0]
        return para_start

    def create_sub_paragraphs(self):
        para_start = self.get_paragraph_start()
        for paragraph in self.paragraphs:
            original_lines = paragraph.get_lines()
            new_main_lines = []
            i = 0

            while i < len(original_lines):
                line = original_lines[i]
                page = self.pages[line.get_page_number()]
                min_x = page.get_min_x()
                max_x = page.get_max_x()
                line_bbox = line.get_line_bbox()
                left_indent = int(line.get_line_bbox().x0) - int(page.get_min_x())
                right_indent = int(page.get_max_x()) - int(line.get_line_bbox().x1)

                is_sub_para = False
                layout_is_centered = abs(left_indent - right_indent) <= 1

                if i > 0 and line.get_font_size() > 9:
                    # Left-aligned logic
                    if ((int(line_bbox.x0) == int(para_start) and int(line_bbox.x0) != int(min_x)) or (
                            int(line_bbox.x0) > int(para_start))) and not layout_is_centered:
                        is_sub_para = True

                    #  Center-aligned logic
                    elif layout_is_centered and left_indent != 0 and right_indent != 0:
                        is_sub_para = False

                    #  Right-aligned logic
                    elif int(line_bbox.x0) != int(para_start) and int(line_bbox.x0) != int(min_x) and int(
                            line_bbox.x1) == int(max_x):
                        is_sub_para = True

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
        extracted_page_number = None
        page_number_start = None

        for page in self.pages:
            if page.get_extracted_page_number() and extracted_page_number is None:
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

    def is_centered(self, paragraph, page_number):
        page = self.pages[page_number]
        left_indent = int(paragraph.get_para_bbox().x0) - int(page.get_min_x())
        right_indent = int(page.get_max_x()) - int(paragraph.get_para_bbox().x1)

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

        for paragraph in self.paragraphs:
            paragraph.set_volume(volume_name)

    def add_chapters(self):
        drawings = self._get_non_footer_drawings()
        page_to_paragraphs = self._group_paragraphs_by_page()
        chapter_names = self._extract_chapter_names(drawings, page_to_paragraphs)

        if chapter_names:
            self._assign_chapters_to_paragraphs(chapter_names)

    def _group_paragraphs_by_page(self):
        """Groups paragraphs by their page number."""
        page_dict = defaultdict(list)
        for paragraph in self.paragraphs:
            page_number = paragraph.get_page_number()
            page_dict[page_number].append(paragraph)
        return page_dict

    def _extract_chapter_names(self, drawings, page_dict):
        """Extracts chapter names that appear immediately after drawing elements."""
        chapters = []
        for drawing in drawings:
            page_number = drawing.get_page_number()
            bbox = drawing.get_bbox()
            paragraphs = page_dict.get(page_number, [])

            for paragraph in paragraphs:
                if paragraph.get_para_bbox().y0 >= bbox.y1:
                    if self.is_centered(paragraph, page_number):
                        chapter_text = " ".join(line.get_text() for line in paragraph.get_lines())
                        chapters.append(chapter_text)
                    break  # Only check the first paragraph below the drawing
        return chapters

    def _assign_chapters_to_paragraphs(self, chapter_names):
        """Assigns the appropriate chapter name to each paragraph and its sub-paragraphs."""
        current_chapter = None
        for paragraph in self.paragraphs:
            para_text = " ".join(line.get_text() for line in paragraph.get_lines())
            if para_text in chapter_names:
                current_chapter = para_text

            paragraph.set_chapter(current_chapter)
            for sub_para in paragraph.get_sub_paragraphs():
                sub_para.set_chapter(current_chapter)


    def process_document(self):
        print('[INFO] Processing Document')
        for page in self.pages:
            print(
                f"Page {page.get_page_number()} Dimensions:\n"
                f"Min X: {page.get_min_x()}\n"
                f"Max X: {page.get_max_x()}\n"
                f"Min Y: {page.get_min_y()}\n"
                f"Max Y: {page.get_max_y()}"
            )
            page.process_page()

        self.paragraphs = [para for page in self.pages for para in page.get_paragraphs()]
        if self.paragraphs:
            self.merge_paragraphs()
            self.create_sub_paragraphs()
            self.add_chapters()
            self.add_volume()