import re
import sys
from typing import Tuple, List, Dict

import os
import fitz
import torch
from IndicTransToolkit.processor import IndicProcessor
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from fitz import Rect
from fontTools.ttLib import TTFont as FontToolsTTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont as ReportlabTTFont
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig

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
                    # num_beams=7,
                    num_beams=4,
                    length_penalty=1.0,
                    early_stopping=True,
                    repetition_penalty=1.1,
                    do_sample=True,
                    # temperature = 0.8,
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
    def _parse_span(span: dict, page_num: int) -> Dict:
        """
        Parses a span and returns cleaned span information.
        Converts ALL-UPPERCASE spans to Title Case (e.g., 'PREFACE' -> 'Preface', 'SIR B.N. RAU' -> 'Sir B.N. Rau').
        """
        text = span["text"].strip()
        if not text:
            return {}

        font = span["font"]

        # Detect if the text is ALL CAPS and contains at least one letter
        if text.isupper() and any(c.isalpha() for c in text):
            text = text.title()

        # Preserve bold styling if present in font name
        if "bold" in font.lower():
            text = f"[3001]{text}[4001]"

        return {
            "text": text,
            "font": font,
            "size": span["size"],
            "page_num": page_num,
            "origin": span["origin"],
            "bbox": fitz.Rect(span["bbox"]),
        }

    def extract_pages(self, pdf_path: str) -> List[Page]:
        doc = fitz.open(pdf_path)
        pages = []

        for page in doc:
            blocks = page.get_text("dict", sort=True)["blocks"]
            page_num = page.number
            pg = Page(number=page_num)
            pg.add_drawings(page.get_cdrawings())
            min_x, max_x, min_y, max_y = self.get_content_dimensions(blocks)
            pg.set_content_dimensions(min_x, max_x, min_y, max_y)

            for block in blocks:
                if block["type"] != 0:
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        parsed = self._parse_span(span, page_num)
                        if parsed:
                            pg.add_span(parsed)

            pages.append(pg)

        return pages

    def translate_preserve_styles(self,paragraph, tgt_lang):
        paragraph_bbox = paragraph.get_para_bbox()

        # 1. Apply invisible style markers
        text = " ".join(line.get_text() for line in paragraph.get_lines())

        # Translate with context
        translated = self.translate_text(text, tgt_lang)

        text = " ".join(translated)

        print(f'[INFO] Translated Text: {text}')

        new_line = Line(
            page_number=paragraph.get_page_number(),
            text=text,
            bbox=paragraph_bbox,
            font_size=paragraph.get_font_size()
        )

        paragraph.set_lines([new_line])

        return paragraph

    @staticmethod
    def parse_styled_spans(text):
        """
        Parses the HTML-like text (<b>, <i>) and returns a list of spans.
        Each span is a dict with keys: text, bold, italic.
        """
        spans = []
        stack = []
        buffer = ""

        def flush_buffer():
            nonlocal buffer
            if buffer:
                style = {"bold": "b" in stack}
                spans.append({"text": buffer.strip(), **style})
                buffer = ""

        tokens = re.split(r"(<\/?b>)", text)

        for token in tokens:
            if token == "":
                continue
            if token in "<b>":
                flush_buffer()
                stack.append(token[1])  # add 'b'
            elif token in "</b>":
                flush_buffer()
                tag = token[2]
                if tag in stack:
                    stack.remove(tag)
            else:
                buffer += token

        flush_buffer()
        return spans

    def layout_paragraph(self, lines):
        """
        Prepare styled word runs for insertion into DOCX.
        No need to calculate line breaks — Word handles wrapping.
        """

        text = " ".join(line.get_text() for line in lines)

        styled_spans = self.parse_styled_spans(text)
        runs = []

        for span in styled_spans:
            for word in span["text"].split():
                runs.append({
                    "text": word,
                    "bold": span.get("bold", False),
                })

        return runs

    def _write_paragraph_to_docx(self, docx_doc, page, runs, paragraph):
        para = docx_doc.add_paragraph()
        self.set_rtl(para)
        para_format = para.paragraph_format

        left_indent = int(paragraph.get_start()) - int(page.get_min_x())
        right_indent = int(page.max_x) - int(paragraph.get_end())

        if left_indent!= right_indent:
            para.alignment = WD_ALIGN_PARAGRAPH.LEFT
            para_format.first_line_indent = Pt(paragraph.get_font_size() * self.language_config.get_font_multiplier() * 2)
        else:
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            para_format.first_line_indent = Pt(0)
        para_format.space_after = Pt(paragraph.get_font_size() * 0.5)

        para_format.line_spacing = Pt(
            paragraph.get_font_size() *
            self.language_config.get_font_multiplier() *
            self.language_config.get_line_spacing_multiplier()
        )

        para_format.space_before = Pt(0)

        for run_info in runs:
            run = para.add_run(run_info["text"] + " ")
            run.font.size = Pt(paragraph.get_font_size() * self.language_config.get_font_multiplier())
            run.font.name = self.font_name
            r = run._element
            r.rPr.rFonts.set(qn('w:eastAsia'), self.font_name)

            if run_info.get("bold"):
                run.bold = True

    @staticmethod
    def _configure_docx_section(docx_doc):
        section = docx_doc.sections[0]

        # Set A4 size (default, but making it explicit)
        section.page_height = Inches(11.69)  # 297 mm
        section.page_width = Inches(8.27)  # 210 mm

        # Set margins to center a 140mm x 222mm area (5.51in x 8.74in)
        section.top_margin = Inches(1.48)  # (297 - 222)/2 mm
        section.bottom_margin = Inches(1.48)
        section.left_margin = Inches(1.38)  # (210 - 140)/2 mm
        section.right_margin = Inches(1.38)

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
                new_para.set_para_bbox(None)  # Optional: compute merged bbox if needed
                new_para.set_start(paragraph.get_start())
                new_para.set_end(prev_paragraph.get_end())
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

    def process_pdf(self, input_folder_path: str,output_folder_path: str):
        # 1. Extract text with styling
        pages = self.extract_pages(input_folder_path)

        docx_doc = Document()
        self._configure_docx_section(docx_doc)

        paragraphs = []

        for page in pages:
            page.process_page()
            paragraphs.extend(page.get_paragraphs())

        # merge paragraphs
        merged_paragraphs = self.merge_paragraphs(paragraphs, pages)

        print(f"[INFO] Merged Paragraphs: {merged_paragraphs}")

        for paragraph in merged_paragraphs:
            translated_para = self.translate_preserve_styles(paragraph, self.target_language_key)

            runs = self.layout_paragraph(lines = translated_para.get_lines())

            self._write_paragraph_to_docx(docx_doc, pages[paragraph.get_page_number()], runs, paragraph)

        file_name = os.path.splitext(os.path.basename(input_folder_path))[0]
        output_docx_path = os.path.join(output_folder_path, "{}.docx".format(file_name+'_'+self.language_config.get_target_language()))
        docx_doc.save(output_docx_path)
           
            # TODO:
            # 1. ADD MARKERS FOR NEW LINE
            # 2. CHECK OUTPUT FOR MULTIPLE PAGES
            # 3. ADD TABS FOR LINES WITHIN THE SAME PARAGRAPHS, AS SOME TEXTS IN THE SAME PARAGRAPHS ARE EITHER CENTERED OR LEFT ALIGNED OR RIGHT ALIGNED,
            # OR HAVE SOME TABS, SO LOOK INTO HOW I CAN MAINTAIN THE FORMAT IN THE TRANSLATED DOCUMENT
            # 4. FOR MULTIPLE PAGES, CHECK LINE SPACING(AND DO WE NEED TO HANDLE LINE SPACING DIFFERENTLY FOR DIFFERENT PAGES)
            # 5. HANDLE HEADERS AND FOOTERS DIFFERENTLY
            # 6. MANAGE FORMAT BETTER

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
            


        #
        #

