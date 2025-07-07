import re
import sys
from typing import Tuple, List
from collections import Counter

import os
import fitz
import torch
from IndicTransToolkit.processor import IndicProcessor
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from fitz import Rect
from fontTools.ttLib import TTFont as FontToolsTTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont as ReportlabTTFont
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig

from src.model.footer import Footer
from src.model.span import Span
from src.model.page import Page
from src.model.paragraph import Paragraph
from src.model.line import Line
from src.model.language_config import LanguageConfig
from src.utils.utils import Utils


# TAGS I CAN USE [B.O.L.D]-> INDICTRANS DOESNT TRANSLATE THIS


class PDFTranslator:
    BATCH_SIZE = 10
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CKPT_DIR = 'ai4bharat/indictrans2-en-indic-1B'

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

            return final

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
        paragraph_bbox = paragraph.get_para_bbox()

        # 1. Apply invisible style markers
        text = " ".join(line.get_text() for line in paragraph.get_lines())

        # Translate with context
        translated = self.translate_text(text, self.target_language_key)

        text = " ".join(translated)

        print(f'[INFO] Translated Text: {text}')

        new_line = Line(
            page_number=paragraph.get_page_number(),
            text=text,
            bbox=paragraph_bbox,
            font_size=paragraph.get_font_size()
        )

        paragraph.set_lines([new_line])

        if paragraph.get_sub_paragraphs():
            sub_para = []
            for para in paragraph.get_sub_paragraphs():
                sub_para.append(self.translate_preserve_styles(para))
            paragraph.set_sub_paragraphs(sub_para)

        if paragraph.get_footer():
            translated_footer=[]
            for footer in paragraph.get_footer():
                translated_text = self.translate_text(footer.get_text(), self.target_language_key)
                translated_footer.append(Footer(text= translated_text))
            paragraph.set_footers(translated_footer)
        return paragraph

    @staticmethod
    def parse_styled_spans(text):
        """
        Parses HTML-like text (e.g., <b>, <n>) into a list of spans.
        Each span is a dict with keys: text, bold, newline.
        """
        spans = []
        stack = []
        buffer = ""

        def flush_buffer():
            nonlocal buffer
            if buffer.strip():
                style = {
                    "bold": "b" in stack,
                }
                spans.append({"text": buffer.strip(), **style})
            buffer = ""

        # Updated regex to capture <b>, </b>
        tokens = re.split(r'(<\/?b>)', text)

        for token in tokens:
            if not token:
                continue
            if token == "<b>":
                flush_buffer()
                stack.append("b")
            elif token == "</b>":
                flush_buffer()
                if "b" in stack:
                    stack.remove("b")
            else:
                buffer += token

        flush_buffer()
        return spans

    def build_document(self, docx_doc, page, paragraph, para_start):
        section = docx_doc.sections[-1]
        new_para = docx_doc.add_paragraph()
        self.set_rtl(new_para)

        self._set_paragraph_alignment_and_indent(new_para, paragraph, page, section, para_start)
        self._set_paragraph_spacing(new_para, paragraph)

        translated_text = " ".join(line.get_text() for line in paragraph.get_lines())
        self._add_text_with_styling(new_para, translated_text, paragraph)

        if paragraph.get_footer():
            self._add_footer(docx_doc, paragraph.get_footer())

        self._process_sub_paragraphs(docx_doc, paragraph, section, page, para_start)

    @staticmethod
    def _set_paragraph_alignment_and_indent(paragraph_obj, paragraph_data, page, section, para_start):
        para_format = paragraph_obj.paragraph_format
        left_indent = int(paragraph_data.get_start()) - int(page.get_min_x())
        right_indent = int(page.get_max_x()) - int(paragraph_data.get_end())

        if abs(left_indent - right_indent) > 1:
            if int(paragraph_data.get_start())!=int(para_start) and int(paragraph_data.get_para_bbox().x0)!=page.get_min_x() and int(paragraph_data.get_end())==int(page.get_max_x()):
                paragraph_obj.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                para_format.first_line_indent = Inches(0)
            else:
                paragraph_obj.alignment = WD_ALIGN_PARAGRAPH.LEFT
                bbox_x0_in = int(paragraph_data.get_para_bbox().x0) / 72
                relative_indent_in = (bbox_x0_in - section.left_margin.inches)
                relative_indent_in = 0 if int(paragraph_data.get_para_bbox().x0)==int(page.get_min_x()) else relative_indent_in
                para_format.first_line_indent = Inches(relative_indent_in)
        else:
            paragraph_obj.alignment = WD_ALIGN_PARAGRAPH.CENTER
            para_format.first_line_indent = Pt(0)

    def _set_paragraph_spacing(self, paragraph_obj, paragraph_data):
        para_format = paragraph_obj.paragraph_format
        para_format.space_after = Pt(paragraph_data.get_font_size() * 0.5)
        para_format.line_spacing = Pt(
            paragraph_data.get_font_size() *
            self.language_config.get_font_multiplier() *
            self.language_config.get_line_spacing_multiplier()
        )
        para_format.space_before = Pt(0)

    def _add_text_with_styling(self, paragraph_obj, text, paragraph_data):
        spans = self.parse_styled_spans(text)
        for span in spans:
            run = paragraph_obj.add_run(span["text"] + " ")
            run.font.size = Pt(paragraph_data.get_font_size() * self.language_config.get_font_multiplier())
            run.font.name = self.font_name
            run._element.rPr.rFonts.set(qn('w:eastAsia'), self.font_name)
            if span.get("bold"):
                run.bold = True

    def _process_sub_paragraphs(self, docx_doc, paragraph_data, section, page, para_start):
        if not paragraph_data.get_sub_paragraphs():
            return

        for sub_para in paragraph_data.get_sub_paragraphs():
            new_para = docx_doc.add_paragraph()
            self.set_rtl(new_para)

            para_format = new_para.paragraph_format

            if int(paragraph_data.get_start()) != int(para_start) and int(paragraph_data.get_end())==int(page.get_max_x()) and int(paragraph_data.get_para_bbox().x0)!=page.get_min_x():
                new_para.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                para_format.first_line_indent = Inches(0)
            else:
                new_para.alignment = WD_ALIGN_PARAGRAPH.LEFT
                bbox_x0_in = int(sub_para.get_para_bbox().x0) / 72
                relative_indent_in = bbox_x0_in - section.left_margin.inches
                relative_indent_in = 0 if int(paragraph_data.get_para_bbox().x0)==int(page.get_min_x()) else relative_indent_in
                para_format.first_line_indent = Inches(relative_indent_in)

            para_format.line_spacing = Pt(
                paragraph_data.get_font_size() *
                self.language_config.get_font_multiplier() *
                self.language_config.get_line_spacing_multiplier()
            )
            para_format.space_before = Pt(0)
            para_format.space_after = Pt(0)

            translated_text = " ".join(line.get_text() for line in sub_para.get_lines())
            self._add_text_with_styling(new_para, translated_text, paragraph_data)

            if sub_para.get_footer():
                self._add_footer(docx_doc,sub_para.get_footer())

    @staticmethod
    def _configure_docx_section(docx_doc):
        section = docx_doc.sections[-1]

        # Set A4 size (default, but making it explicit)
        section.page_height = Inches(11.69)  # 297 mm
        section.page_width = Inches(8.27)  # 210 mm

        # Set margins to center a 140mm x 222mm area (5.51in x 8.74in)
        section.top_margin = Inches(1.48)  # (297 - 222)/2 mm
        section.bottom_margin = Inches(1.48)
        section.left_margin = Inches(1.38)  # (210 - 140)/2 mm
        section.right_margin = Inches(1.38)

    def create_sub_paragraphs(self, merged_paragraphs, pages, para_start):
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
        bottom_border = OxmlElement('w:bottom')
        bottom_border.set(qn('w:val'), 'single')  # Line style
        bottom_border.set(qn('w:sz'), '6')  # Line width (8ths of a point)
        bottom_border.set(qn('w:space'), '0')  # Space above line
        bottom_border.set(qn('w:color'), '000000')  # Black color

        p_bdr.append(bottom_border)

    # TODO: FIX THE FOOTER FUNCTION, IT IS ADDING FOOTER TO ALL THE PAGES
    def _add_footer(self,docx_doc, para_footers: List[Footer]):
        """Adds footer ONLY to the current section's next page"""
        # 1. Get current section (always use last section)
        section = docx_doc.sections[-1]
        footer = section.footer

        # 2. CRITICAL: Disable footer inheritance
        footer.is_linked_to_previous = False

        # 3. Add footer content
        line_para = footer.add_paragraph()
        line_para.alignment = WD_ALIGN_PARAGRAPH.LEFT
        line_para.paragraph_format.left_indent = Pt(0)
        line_para.paragraph_format.right_indent = Pt(0)
        # TODO: EVERYTIME WE ARE ADDING THE FOOTER WE ARE ADDING A LINE, CHECK IF THE SECTION THAT I AM ADDING
        # THE FOOTER TO HAS A LINE OR NOT, IF IT DOES NOT ONLY THEN ADD THE LINE
        self.add_horizontal_line(line_para)

        for para_footer in para_footers:
            footer_para = footer.add_paragraph()
            footer_para.alignment = WD_ALIGN_PARAGRAPH.LEFT
            footer_format = footer_para.paragraph_format
            footer_format.space_before = Pt(0)
            footer_format.space_after = Pt(0)
            run = footer_para.add_run(para_footer.get_text())
            run.font.size = Pt(para_footer.get_font_size() * self.language_config.get_font_multiplier())

        # 5. Ensure content stays with footer
        if docx_doc.paragraphs:  # If paragraphs exist
            docx_doc.paragraphs[-1].paragraph_format.keep_with_next = True


    def process_pdf(self, input_folder_path: str,output_folder_path: str):
        # 1. Extract text with styling
        pages = self.extract_pages(input_folder_path)

        docx_doc = Document()
        self._configure_docx_section(docx_doc)

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
            self.build_document(docx_doc, pages[paragraph.get_page_number()], paragraph, para_start)

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



