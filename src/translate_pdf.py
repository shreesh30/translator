import os
import re
import sys
from collections import Counter
from typing import Tuple, List, Dict

import fitz
import torch
from IndicTransToolkit.processor import IndicProcessor
from PIL import ImageFont
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt
from fontTools.ttLib import TTFont as FontToolsTTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont as ReportlabTTFont
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig

from src.model.footer import Footer
from src.model.language_config import LanguageConfig
from src.model.line import Line
from src.model.page import Page
from src.model.paragraph import Paragraph
from src.model.span import Span
from src.utils.utils import Utils


class PDFTranslator:
    BATCH_SIZE = 10
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CKPT_DIR = 'ai4bharat/indictrans2-en-indic-1B'

    # PAGE DIMENSIONS
    STANDARDIZED_PAGE_HEIGHT =Inches(11.69)
    STANDARDIZED_PAGE_WIDTH = Inches(8.27)
    STANDARDIZED_TOP_MARGIN = Inches(1.48)
    STANDARDIZED_BOTTOM_MARGIN = Inches(1.38)
    STANDARDIZED_LEFT_MARGIN = Inches(1.38)
    STANDARDIZED_RIGHT_MARGIN = Inches(1.38)
    STANDARDIZED_FOOTER_DISTANCE = Inches(1)

    USABLE_PAGE_HEIGHT = (STANDARDIZED_PAGE_HEIGHT.inches- STANDARDIZED_TOP_MARGIN.inches- STANDARDIZED_BOTTOM_MARGIN.inches)  # in inches
    USABLE_PAGE_WIDTH = (STANDARDIZED_PAGE_WIDTH.inches - STANDARDIZED_LEFT_MARGIN.inches - STANDARDIZED_RIGHT_MARGIN.inches)
    PAGE_USED = 0

    def __init__(self,lang_config:LanguageConfig, quantization=None):
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')

        self.quantization = quantization
        self.tokenizer= self.initialize_tokenizer(self.CKPT_DIR)
        self.processor = IndicProcessor(inference=True)
        # self.config = AutoConfig.from_pretrained(self.CKPT_DIR)
        self.model = self.initialize_model(self.CKPT_DIR)

        self.language_config = lang_config
        self.target_language = self.language_config.get_target_language()
        self.target_language_key = self.language_config.get_target_language_key()
        self.source_language = self.language_config.get_source_language()
        self.source_language_key = self.language_config.get_source_language_key()

        self.font_name = self.load_and_register_font(self.language_config.get_target_font_path())

    @staticmethod
    def load_and_register_font(font_path):
        try:
            tt = FontToolsTTFont(font_path)
            font_name = tt['name'].getDebugName(1).split('-')[0]
            tt.close()
            print(f"Detected font name: {font_name}")
        except Exception as e:
            raise RuntimeError(f"Font metadata read failed: {str(e)}")

        try:
            pdfmetrics.registerFont(ReportlabTTFont(font_name, font_path))
            print(f"Successfully registered font: {font_name}")
        except Exception as e:
            raise RuntimeError(f"Font registration failed: {str(e)}")

        return font_name

    @staticmethod
    def initialize_tokenizer(ckpt_dir):
        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir,trust_remote_code=True)
        return tokenizer

    def initialize_model(self, ckpt_dir):
        if self.quantization == "4-bit":
            qconfig = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        elif self.quantization == "8-bit":
            qconfig = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_use_double_quant=True,
                bnb_8bit_compute_dtype=torch.bfloat16,
            )
        else:
            qconfig = None

        model = AutoModelForSeq2SeqLM.from_pretrained(
            ckpt_dir,
            trust_remote_code=True,
        )

        if qconfig is None:
            model = model.to(self.DEVICE)
            if self.DEVICE == "cuda":
                model.half()

        model.eval()

        return model


    def translate_text(self, text: str, tgt_lang: str) -> str:
        print(f"[INFO] Translating single text to {tgt_lang}...")
        print(f"[INFO] Translating the text:")
        print(text)
        try:
            processed = self.processor.preprocess_batch([text], src_lang=self.source_language_key, tgt_lang=tgt_lang)
            print(f"[DEBUG] Preprocessed: {processed}")

            inputs = self.tokenizer(processed, truncation=True, padding=True, return_tensors="pt").to(
                self.DEVICE)
            print(f"[DEBUG] Tokenized Inputs: {inputs}")

            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    min_length=0,
                    max_length=512,
                    num_beams=6,
                    length_penalty=1.0,
                    early_stopping=True,
                    repetition_penalty=1.1,
                    do_sample=True,
                    temperature = 0.7,
                    no_repeat_ngram_size=3
                )

                print(f"[DEBUG] Generated Token IDs: {output}")
            with self.tokenizer.as_target_tokenizer():
                decoded = self.tokenizer.batch_decode(
                    output.detach().cpu().tolist(),
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )

                print(f"[DEBUG] Decoded Test: {decoded}")

            final = self.processor.postprocess_batch(decoded, lang=tgt_lang)
            print(f"[DEBUG] Postprocessed Translations: {final}")
            print(f"[DEBUG] Translated: {final[0]}")

            # Define tag replacements


            # Apply replacements
            for old, new in Utils.tag_map.items():
                final[0] = final[0].replace(old, new)

            return final[0]

        except Exception as e:
            print(f"[ERROR] Translation failed: {e}")
            return ""

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
                    min_y = min(min_y,y0)
                    max_y = max(max_y, y1)

        if min_x == float('inf') or max_x == float('-inf') or min_y== float('inf') or max_y == float('-inf'):
            return 0, 0, 0 ,0  # No text found

        return min_x, max_x, min_y, max_y

    @staticmethod
    def _parse_span(span: dict, page_num: int):
        """
        Parses a span and returns cleaned span information.
        Converts ALL-UPPERCASE spans to Title Case (e.g., 'PREFACE' -> 'Preface', 'SIR B.N. RAU' -> 'Sir B.N. Rau').
        """
        text = span["text"].strip()
        if not text:
            return {}

        font = span["font"]

        # Preserve bold styling if present in font name
        if "bold" in font.lower():
            text = f"[123] {text} [456]"

        new_span = Span()
        new_span.set_text(text)
        new_span.set_font(font)
        new_span.set_font_size(span["size"])
        new_span.set_page_num(page_num)
        new_span.set_origin(span["origin"])
        new_span.set_bbox(fitz.Rect(span["bbox"]))

        return new_span

    def extract_pages(self, pdf_path: str) -> List[Page]:
        doc = fitz.open(pdf_path)
        pages = []

        for page in doc:
            blocks = page.get_text("dict", sort=True)["blocks"]
            page_num = page.number
            pg = Page(number=page_num, target_language= self.target_language)
            pg.add_drawings(page.get_cdrawings())
            min_x, max_x, min_y, max_y = self.get_content_dimensions(blocks)
            pg.set_content_dimensions(min_x, max_x, min_y, max_y)

            for block in blocks:
                if block["type"] != 0:
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        new_span = self._parse_span(span, page_num)
                        pg.add_span(new_span)

            pages.append(pg)

        return pages

    def translate_preserve_styles(self,paragraph):
        # 1. Apply invisible style markers
        text = " ".join(line.get_text() for line in paragraph.get_lines())

        # Translate with context
        translated_text = self.translate_text(text, self.target_language_key)

        print(f'[INFO] Translated Text: {translated_text}')

        lines = self.split_into_lines(translated_text, paragraph.get_font_size())

        new_lines=[]
        for line in lines:
            new_line = self.convert_tags_to_angle_brackets(line)
            line_bbox = self.get_line_bbox(line, paragraph.get_font_size())
            new_lines.append(Line(page_number=paragraph.get_page_number(),
                                  text=new_line,
                                  line_bbox= line_bbox,
                                  bbox=line_bbox,
                                  font_size=paragraph.get_font_size()))

        paragraph.set_lines(new_lines)

        if paragraph.get_sub_paragraphs():
            sub_para = []
            for para in paragraph.get_sub_paragraphs():
                sub_para.append(self.translate_preserve_styles(para))
            paragraph.set_sub_paragraphs(sub_para)

        if paragraph.get_footer():
            translated_footer=[]
            for footer in paragraph.get_footer():
                translated_text = self.translate_text(footer.get_text(), self.target_language_key)
                translated_footer.append(Footer(text= translated_text, font_size=footer.get_font_size()))
            paragraph.set_footers(translated_footer)
        return paragraph

    @staticmethod
    def parse_styled_spans(text)->List[Dict]:
        """
        Parses text and marks any content inside angle brackets <...> as bold.
        Returns a list of spans: [{"text": ..., "bold": True/False}]
        """
        spans = []
        pattern = re.compile(r'<(.*?)>')

        last_index = 0

        for match in pattern.finditer(text):
            start, end = match.span()

            # Add normal text before the tag
            if start > last_index:
                normal_text = text[last_index:start].strip()
                if normal_text:
                    spans.append({"text": normal_text, "bold": False})

            # Add bold text inside angle brackets
            bold_text = match.group(1).strip()
            if bold_text:
                spans.append({"text": bold_text, "bold": True})

            last_index = end

        # Add any remaining text after the last tag
        if last_index < len(text):
            tail = text[last_index:].strip()
            if tail:
                spans.append({"text": tail, "bold": False})

        return spans

    @staticmethod
    def clear_footer(section):
        """Completely eradicates all footer content and settings"""
        # Clear primary footer
        footer = section.footer
        for elem in list(footer._element):
            footer._element.remove(elem)

        # Clear first page footer if exists
        if hasattr(section, 'first_page_footer'):
            first_footer = section.first_page_footer
            for elem in list(first_footer._element):
                first_footer._element.remove(elem)

        # Clear even page footer if exists
        if hasattr(section, 'even_page_footer'):
            even_footer = section.even_page_footer
            for elem in list(even_footer._element):
                even_footer._element.remove(elem)


    def build_document(self, docx_doc, page, paragraph, para_start):
        lines = [line.get_text() for line in paragraph.get_lines()]
        current_para = None

        has_footer = bool(paragraph.get_footer())
        # Flag to track if footer for current paragraph has been added to a section
        footer_added = False

        for i,line in enumerate(lines):
            # Calculate height of current line
            line_height = self.estimate_line_height(line, paragraph)
            line_spacing = self.language_config.get_line_spacing_multiplier()* self.language_config.get_font_size_multiplier()*paragraph.get_font_size()

            final_line_height = (line_height+line_spacing)/72 if i!=len(lines)-1 else line_height/72
            space_used = self.PAGE_USED + final_line_height

            # Check if we need to create a new section (page) or paragraph
            if space_used > self.USABLE_PAGE_HEIGHT:
                print('[INFO] Creating New Section')
                # If this is the last para in the section, remove space after for it before creating a new section
                if docx_doc.paragraphs:
                    current_para_format = docx_doc.paragraphs[-1].paragraph_format
                    current_para_format.space_after = Pt(0)

                # If footer is pending for current paragraph, attach to current_section BEFORE making a new one
                if has_footer and not footer_added:
                    self._add_footer(docx_doc.sections[-1], paragraph.get_footer())
                    footer_added = True

                # Create new section (page)
                docx_doc.add_section(WD_SECTION.NEW_PAGE)
                self._configure_docx_section(docx_doc.sections[-1])
                self.PAGE_USED = 0
                current_para = None  # Force new paragraph on new page

            # Create new paragraph if needed (first line or line doesn't fit in current para)
            if current_para is None:
                print('[INFO] Creating New Paragraph')
                current_para = docx_doc.add_paragraph()
                self.set_rtl(current_para)
                self._set_paragraph_alignment_and_indent(current_para, paragraph, page, para_start, docx_doc.sections[-1])
                self._set_paragraph_spacing(current_para, paragraph)

                if has_footer and not footer_added:
                    self._add_footer(docx_doc.sections[-1], paragraph.get_footer())
                    footer_added =True

            # Add the line to current paragraph
            self._add_text_with_styling(current_para, line, paragraph)
            self.PAGE_USED += line_height/72

        if has_footer and not footer_added:
            self._add_footer(docx_doc.sections[-1], paragraph.get_footer())


        self.PAGE_USED += ((paragraph.get_font_size() * 0.5) / 72) # Adding to incorporate "Space After" after a paragraph

        # Process sub-paragraphs
        if paragraph.get_sub_paragraphs():
            for para in paragraph.get_sub_paragraphs():
                self.build_document(docx_doc, page, para, para_start)

    @staticmethod
    def _set_paragraph_alignment_and_indent(paragraph_obj, paragraph_data, page, para_start, section, continue_para=False):
        para_format = paragraph_obj.paragraph_format
        left_indent = int(paragraph_data.get_start()) - int(page.get_min_x())
        right_indent = int(page.get_max_x()) - int(paragraph_data.get_end())

        if abs(left_indent - right_indent) > 0:
            if int(paragraph_data.get_start())!=int(para_start) and int(paragraph_data.get_para_bbox().x0)!=page.get_min_x() and int(paragraph_data.get_end())==int(page.get_max_x()):
                paragraph_obj.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                para_format.first_line_indent = Inches(0)
            else:
                paragraph_obj.alignment = WD_ALIGN_PARAGRAPH.LEFT
                relative_indent_in=0
                if not continue_para:
                    bbox_x0_in = int(paragraph_data.get_para_bbox().x0) / 72
                    relative_indent_in = (bbox_x0_in - section.left_margin.inches)
                    if int(paragraph_data.get_para_bbox().x0)==int(page.get_min_x()):
                        relative_indent_in = 0
                    elif int(paragraph_data.get_para_bbox().x0)==int(para_start):
                        relative_indent_in=relative_indent_in/2
                para_format.first_line_indent = Inches(relative_indent_in)
        else:
            paragraph_obj.alignment = WD_ALIGN_PARAGRAPH.CENTER
            para_format.first_line_indent = Pt(0)

    def _set_paragraph_spacing(self, paragraph_obj, paragraph_data):
        para_format = paragraph_obj.paragraph_format
        para_format.space_after = Pt(paragraph_data.get_font_size() * 0.5)
        para_format.line_spacing = Pt(
            paragraph_data.get_font_size() *
            self.language_config.get_font_size_multiplier() *
            self.language_config.get_line_spacing_multiplier()
        )
        para_format.space_before = Pt(0)

    def _add_text_with_styling(self, paragraph_obj, text, paragraph_data):
        spans = self.parse_styled_spans(text)
        for span in spans:
            run = paragraph_obj.add_run(span["text"] + " ")
            run.font.size = Pt(paragraph_data.get_font_size() * self.language_config.get_font_size_multiplier())
            run.font.name = self.font_name
            run._element.rPr.rFonts.set(qn('w:eastAsia'), self.font_name)
            if span.get("bold"):
                run.bold = True


    def _configure_docx_section(self,section):
        # Set standard page dimensions and margins
        section.page_height = self.STANDARDIZED_PAGE_HEIGHT
        section.page_width = self.STANDARDIZED_PAGE_WIDTH
        section.top_margin = self.STANDARDIZED_TOP_MARGIN
        section.bottom_margin = self.STANDARDIZED_BOTTOM_MARGIN
        section.left_margin = self.STANDARDIZED_LEFT_MARGIN
        section.right_margin = self.STANDARDIZED_RIGHT_MARGIN

        section.different_first_page_header_footer = True
        section.footer.is_linked_to_previous = False
        section.header.is_linked_to_previous = False

        return section

    @staticmethod
    def _universal_clear_element(element):
        """Works with any python-docx version"""
        try:
            # Try modern clear() method first
            element.clear()
        except AttributeError:
            # Fallback for older versions
            element._element.clear()  # Nuclear option - removes all XML content

    @staticmethod
    def _clear_section_footer_content(section):
        footer = section.footer  # Get the specific footer object for this section
        for elem in list(footer._element):  # Iterate over a copy to allow modification
            footer._element.remove(elem)
        # Word requires at least one paragraph in a header/footer part
        if not footer.paragraphs:
            footer.add_paragraph()

    @staticmethod
    def create_sub_paragraphs(merged_paragraphs, pages, para_start):
        for paragraph in merged_paragraphs:
            original_lines = paragraph.get_lines()
            new_main_lines = []
            i = 0

            while i < len(original_lines):
                line = original_lines[i]
                page = pages[line.get_page_number()]
                min_x = page.get_min_x()
                max_x = page.get_max_x()
                line_bbox = line.get_line_bbox()
                left_indent = int(line.get_line_bbox().x0) - int(page.get_min_x())
                right_indent = int(page.get_max_x()) - int(line.get_line_bbox().x1)

                is_sub_para = False
                layout_is_centered = abs(left_indent - right_indent) <= 1

                if i > 0 and line.get_font_size() > 9:
                    # Left-aligned logic
                    if ((int(line_bbox.x0) == int(para_start) and int(line_bbox.x0) != int(min_x)) or (int(line_bbox.x0) > int(para_start))) and not layout_is_centered:
                        is_sub_para = True

                    #  Center-aligned logic
                    elif layout_is_centered and left_indent!=0 and right_indent!=0:
                        is_sub_para = False

                    #  Right-aligned logic
                    elif int(line_bbox.x0) != int(para_start) and int(line_bbox.x0) != int(min_x) and int(line_bbox.x1) == int(max_x):
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
                        next_min_x = pages[next_line.get_page_number()].get_min_x()

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

    @staticmethod
    def merge_paragraphs(paragraphs, pages):
        merged_paragraphs = []
        prev_paragraph = None

        for paragraph in paragraphs:
            curr_page_num = paragraph.get_page_number()
            curr_bbox = paragraph.get_para_bbox()
            curr_start_x = int(curr_bbox.x0) if curr_bbox else None
            page_min_x = int(pages[curr_page_num].get_min_x())

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
                new_para.set_para_bbox(fitz.Rect(prev_paragraph.get_para_bbox().x0, prev_paragraph.get_para_bbox().y0,paragraph.get_para_bbox().x1, paragraph.get_para_bbox().y1))
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

        return merged_paragraphs

    def set_rtl(self, paragraph):
        if self.language_config.get_right_to_left():
            """Set the paragraph to Right-to-Left (RTL) direction."""
            p = paragraph._p  # Access the XML <w:p> element
            pPr = p.get_or_add_pPr()
            bidi = OxmlElement('w:bidi')
            bidi.set(qn('w:val'), '1')  # Enable RTL
            pPr.append(bidi)

    @staticmethod
    def add_horizontal_line(paragraph):
        """Adds a proper horizontal line using paragraph border"""
        p_pr = paragraph._p.get_or_add_pPr()

        # Create border element if it doesn't exist
        p_bdr = OxmlElement('w:pBdr')
        p_pr.append(p_bdr)

        # Create bottom border with specifications - now using black color
        bottom_border = OxmlElement('w:top')
        bottom_border.set(qn('w:val'), 'single')  # Line style
        bottom_border.set(qn('w:sz'), '6')  # Line width (6ths of a point)
        bottom_border.set(qn('w:space'), '0')  # Space above line
        bottom_border.set(qn('w:color'), '000000')  # Black color

        p_bdr.append(bottom_border)

    @staticmethod
    def remove_angle_brackets(text: str) -> str:
        """
        Removes surrounding angle brackets like <...> but keeps inner content.
        """
        # Remove one layer of angle brackets if the entire string is wrapped in them
        cleaned = re.sub(r'^<\s*(.*?)\s*>$', r'\1', text)

        return cleaned.strip()

    @staticmethod
    def convert_tags_to_angle_brackets(text: str) -> str:
        # Replace <b>...</b> or <i>...</i> with <...>
        text = re.sub(r'<[^>/]+>(.*?)</[^>]+>', r'<\1>', text)

        return text.strip()

    def estimate_line_height(self,line, paragraph_data):
        # DIFF
        font_size_pt = paragraph_data.get_font_size() * self.language_config.get_font_size_multiplier()

        # Load font
        font = ImageFont.truetype(self.language_config.get_target_font_path(), size=font_size_pt)

        line_copy = str(line)
        self.remove_angle_brackets(line_copy)

        bbox = font.getbbox(line_copy)
        line_height = bbox[3] - bbox[1]
        return line_height

    def split_into_lines(self, para_text, font_size):
        """Splits text into lines using accurate point/inch measurements"""
        # 1. Get font metrics in points
        font_size_pt = font_size * self.language_config.get_font_size_multiplier()

        font = ImageFont.truetype(self.language_config.get_target_font_path(), size=font_size_pt)

            # 2. Calculate max width IN POINTS (1 inch = 72 points)
        max_width_pt = self.USABLE_PAGE_WIDTH * 72

            # 3. Language-aware splitting
        return self._split_words(para_text, font, max_width_pt)

    @staticmethod
    def _split_words(text, font, max_width_pt):
        """Splits space-separated languages using accurate width measurements"""
        words = text.split()
        lines = []
        current_line = []
        current_width = 0  # in points

        space_width = font.getlength(" ")  # Get space width in font units

        for word in words:
            word_width = font.getlength(word)

            # Check if word fits (including space if not first word)
            if current_line:
                total_width = current_width + space_width + word_width
            else:
                total_width = word_width

            if total_width > max_width_pt:
                if current_line:  # Only break if we have content
                    lines.append(" ".join(current_line))
                current_line = [word]
                current_width = word_width
            else:
                if current_line:
                    current_width += space_width
                current_line.append(word)
                current_width += word_width

        if current_line:
            lines.append(" ".join(current_line))

        return lines

    def get_line_bbox(self, line, font_size):
        line_copy = str(line)
        line_copy = self.remove_angle_brackets(line_copy)

        font_size_pt = font_size * self.language_config.get_font_size_multiplier()
        # Load font
        font = ImageFont.truetype(self.language_config.get_target_font_path(), size=font_size_pt)
        bbox = font.getbbox(line_copy)
        return fitz.Rect(bbox)

    @staticmethod
    def section_has_footer(section):
        for para in section.footer.paragraphs:
            if para.text.strip():  # avoid empty string or whitespace-only paragraphs
                return True
        return False

    def _add_footer(self,section, para_footers: List[Footer]):
        """Adds footer ONLY to the current section's next page"""
        # 1. Get current section (always use last section)
        section.footer_distance = self.STANDARDIZED_FOOTER_DISTANCE
        footer = section.first_page_footer
        footer.is_linked_to_previous = False

        for para_footer in para_footers:
            # 2. Add horizontal line only if this is the FIRST footer paragraph
            if not any(p.text.strip() for p in footer.paragraphs):
                line_para = footer.add_paragraph()
                self.add_horizontal_line(line_para)
                line_para.alignment = WD_ALIGN_PARAGRAPH.LEFT
                line_para_format = line_para.paragraph_format
                line_para_format.left_indent = Pt(0)
                line_para_format.right_indent = Pt(0)
                line_para_format.space_before = Pt(0)
                line_para_format.space_after = Pt(0)
                line_para_format.line_spacing = Pt(0)

            if para_footer.get_text().strip():  # Only add if there's actual text
                footer_para = footer.add_paragraph()
                footer_para.alignment = WD_ALIGN_PARAGRAPH.LEFT  # Or whatever alignment is desired
                footer_format = footer_para.paragraph_format
                footer_format.space_before = Pt(0)
                footer_format.space_after = Pt(0)
                footer_format.line_spacing = Pt(para_footer.get_font_size() * self.language_config.get_font_size_multiplier() *self.language_config.get_line_spacing_multiplier())
                run = footer_para.add_run(para_footer.get_text())
                run.font.size = Pt(para_footer.get_font_size() * self.language_config.get_font_size_multiplier())
                run.font.name = self.font_name  # Ensure font consistency
                run._element.rPr.rFonts.set(qn('w:eastAsia'), self.font_name)  # For East Asian fonts


    def process_pdf(self, input_folder_path: str,output_folder_path: str):
        # 1. Extract text with styling
        pages = self.extract_pages(input_folder_path)

        docx_doc = Document()
        section = docx_doc.sections[-1]
        self._configure_docx_section(section)

        paragraphs = []
        starting_page_number = 0
        for page in pages:
            print(
                f"Page {page.get_page_number()} Dimensions:\n"
                f"Min X: {page.get_min_x()}\n"
                f"Max X: {page.get_max_x()}\n"
                f"Min Y: {page.get_min_y()}\n"
                f"Max Y: {page.get_max_y()}"
            )
            page.process_page()
            paragraphs.extend(page.get_paragraphs())

        # merge paragraphs
        merged_paragraphs = self.merge_paragraphs(paragraphs, pages)

        print(f"[INFO] Merged Paragraphs: {merged_paragraphs}")

        para_start_values = [para.get_start() for para in merged_paragraphs]

        # Count occurrences of each start value
        counter = Counter(para_start_values)

        # Get the most common start value
        para_start = counter.most_common(1)[0][0]

        self.create_sub_paragraphs(merged_paragraphs, pages, para_start)

        print(f'[INFO] Finalised Paragraphs: {merged_paragraphs}')

        for paragraph in merged_paragraphs:
            translated_para = self.translate_preserve_styles(paragraph)
            print(f'Translated Paragraph: {translated_para}')
            self.build_document(docx_doc, pages[paragraph.get_page_number()], translated_para, para_start)

        print(f'[INFO] Total Sections: {len(docx_doc.sections)}')

        file_name = os.path.splitext(os.path.basename(input_folder_path))[0]
        output_docx_path = os.path.join(output_folder_path,"{}.docx".format(file_name + '_' + self.language_config.get_target_language()))
        docx_doc.save(output_docx_path)

            # TODO:
            # THE FOOTERS CAN BE ADDED WHERE I AM CURRENTLY ADDING THEM, THE PAGE NUMBERS SHOULD BE CENTER ALIGNED AT THE BOTTOM FOR ALL PAGES
            # 1. ADD TRANSLATED FOOTERS TO THE BOTTOM OF THE PAGE WHEREVER THE PARAGRAPH EXISTS
            # 2. HANDLE TABLES
            # 3. ADD TRANSLATED HEADERS TO THE TOP OF THE PAGE
            # 4. HANDLE HEADERS AND FOOTERS DIFFERENTLY
            # 5. MANAGE FORMAT BETTER
            # 6. FOR THE CONTENTS TABLE, JUST TRANSLATE THE CONTENTS, LEAVE THE PAGE NUMBERS BLANK

        '''Texts like below are appearing as separate paragraphs
                1. The Hon’ble Sri C. Rajagopalachariar.
                2. Dr. B. Pattabhi Sitaramayya.
                3. The Hon’ble Sri T. Prakasam.
                4. The Hon’ble Dewan Bahadur Sir N. Gopalaswami Ayyangar.
                5. Diwan Bahadur Sir Alladi Krishnaswami Ayyar.
                
                First check output of current code, if it doesn't result into something fruitful then go for the below logic
                According to out current logic, these texts will get centered, we dont want that to happen.
                We change the logic to see if the left margin-start of line==right margin-end of line, if that is the
                case then we need to centre the line else we will have to make it stic to the right
            '''



