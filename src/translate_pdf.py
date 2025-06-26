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
from fitz import Rect
from fontTools.ttLib import TTFont as FontToolsTTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont as ReportlabTTFont
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig

from src.model.page import Page
from src.model.language_config import LanguageConfig

BOLD = "[000]"
ITALIC = "[111]"
NEW_LINE="[222]"


class PDFTranslator:
    BATCH_SIZE = 10
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TMP_DOCX = "resource/tmp/docx/temp.intermediate.docx"
    OUTPUT_DOCX = "resource/output/translated.docx"
    CKPT_DIR = 'ai4bharat/indictrans2-en-indic-1B'
    BOLD_TAG='[111]'

    def __init__(self,lang_config:LanguageConfig, quantization=None):
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')

        self.quantization = quantization
        self.tokenizer= self.initialize_tokenizer(self.CKPT_DIR)
        self.processor = IndicProcessor(inference=True)
        # self.config = AutoConfig.from_pretrained(self.CKPT_DIR)
        self.model = self.initialize_model(self.CKPT_DIR)

        self.tag_map = {'[3001]': '<b>','[3002]': '</b>','[3003]': '<i>','[3004]': '</i>', '[3005]':'</n>'}

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


    @staticmethod
    def is_word_only(text):
        return bool(re.search(r'[a-zA-Z]', text))

    def batch_translate(self, elements, tgt_lang):
        contents = [e['content'] for e in elements]
        translations = []
        total = len(contents)
        print(f"[INFO] Translating {total} elements...")

        for i in range(0, total, self.BATCH_SIZE):
            batch = contents[i:i + self.BATCH_SIZE]
            print(f"\n[INFO] Processing batch {i // self.BATCH_SIZE + 1} ({i} to {min(i + self.BATCH_SIZE, total)})")
            try:
                processed = self.processor.preprocess_batch(batch, src_lang=self.source_language_key, tgt_lang=tgt_lang)
                print(f"[DEBUG] Preprocessed: {processed}")

                inputs = self.tokenizer(processed, truncation=True, padding="longest", return_tensors="pt").to(
                    self.DEVICE)
                print(f"[DEBUG] Tokenized Inputs: {inputs}")

                with torch.no_grad():
                    output = self.model.generate(
                        **inputs,
                        min_length=0,
                        max_length=512,
                        # 2316 with num_beams=6
                        # num_beams=7,
                        num_beams=6,
                        # temperature=0.9,
                        temperature=0.8,
                        length_penalty=1.0,
                        early_stopping=True,
                        # no_repeat_ngram_size=3,
                        repetition_penalty=1.2
                    )

                    print(f"[DEBUG] Generated Token IDs: {output}")

                # with self.tokenizer.as_target_tokenizer():
                decoded = self.tokenizer.batch_decode(
                        output.detach().cpu().tolist(),
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False
                    )
                print(f"[DEBUG] Decoded Text: {decoded}")

                final = self.processor.postprocess_batch(decoded, lang=tgt_lang)
                print(f"[DEBUG] Postprocessed Translations: {final}")
                translations.extend(final)

                del inputs
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"[ERROR] Batch failed: {e}")
                translations.extend([""] * len(batch))

        for elem, trans in zip(elements, translations):
            elem['content'] = trans

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
                    num_beams=8,
                    length_penalty=1.0,
                    early_stopping=True,
                    repetition_penalty=1.2,
                    # temperature=0.6
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
            for old, new in self.tag_map.items():
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

    def _parse_span(self, span: dict, page_num: int) -> Dict:
        """
        Parses a span and returns cleaned span information.
        """
        text = span["text"].strip()
        if not text:
            return None

        font = span["font"]
        if "bold" in font.lower():
            text = f"[3001]{text}[3002]"
        elif "italic" in font.lower():
            text = f"[3003]{text}[3004]"

        return {
            "text": text.lower(),
            "font": font,
            "size": span["size"],
            "page_num": page_num,
            "origin": span["origin"],
            "bbox": fitz.Rect(span["bbox"]),
            "line_bbox": fitz.Rect(span["bbox"]),
        }

    def extract_pages(self, pdf_path: str) -> List[Page]:
        doc = fitz.open(pdf_path)
        pages = []

        for page in doc:
            blocks = page.get_text("dict", sort=True)["blocks"]
            page_num = page.number + 1
            pg = Page(page_num)
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

    @staticmethod
    def get_paragraph_bbox(paragraph):
        x0s, y0s, x1s, y1s = [], [], [], []
        for span in paragraph["elements"]:
            rect = span["bbox"]
            x0s.append(rect.x0)
            y0s.append(rect.y0)
            x1s.append(rect.x1)
            y1s.append(rect.y1)
        return Rect(min(x0s), min(y0s), max(x1s), max(y1s))

    def translate_preserve_styles(self,paragraph, tgt_lang):
        paragraph_bbox = self.get_paragraph_bbox(paragraph)

        # 1. Apply invisible style markers
        text = " ".join(span['text'] for span in paragraph["elements"])

        # Translate with context
        translated = self.translate_text(text, tgt_lang)

        return {
            "paragraph": translated,
            "page_num": paragraph["page_num"],
            "bbox": paragraph_bbox
        }

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
                style = {"bold": "b" in stack, "italic": "i" in stack}
                spans.append({"text": buffer.strip(), **style})
                buffer = ""

        tokens = re.split(r"(<\/?[bi]>)", text)

        for token in tokens:
            if token == "":
                continue
            if token in ("<b>", "<i>"):
                flush_buffer()
                stack.append(token[1])  # add 'b' or 'i'
            elif token in ("</b>", "</i>"):
                flush_buffer()
                tag = token[2]
                if tag in stack:
                    stack.remove(tag)
            else:
                buffer += token

        flush_buffer()
        return spans

    def layout_paragraph(self, text, bbox, font_size=12, font_name="Times-Roman", font_path=None):
        font = fitz.Font(fontname=font_name, fontfile=font_path)
        styled_spans = self.parse_styled_spans(text)

        max_width = bbox.x1 - bbox.x0
        line_height = font_size * 1.2
        current_y = bbox.y0
        lines = []
        current_line = []
        current_line_text = ""

        for span in styled_spans:
            span_words = span["text"].split()
            for word in span_words:
                test_line = (current_line_text + " " + word).strip() if current_line else word
                test_width = font.text_length(test_line, fontsize=font_size)

                if test_width <= max_width:
                    current_line.append({                        "text": word,
                        "bold": span["bold"],
                        "italic": span["italic"]
                    })
                    current_line_text = test_line
                else:
                    if current_line:
                        lines.append((current_line, current_y))
                        current_y += line_height
                    current_line = [{
                        "text": word,
                        "bold": span["bold"],
                        "italic": span["italic"]
                    }]
                    current_line_text = word

        if current_line:
            lines.append((current_line, current_y))

        return lines

    def _write_paragraph_to_docx(self, docx_doc, lines, paragraph, index, total):
        para = docx_doc.add_paragraph()
        para_format = para.paragraph_format

        if len(lines) > 1:
            para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            para_format.first_line_indent = Pt(paragraph["font_size"] * self.language_config.get_font_multiplier() * 2)
        else:
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            para_format.first_line_indent = Pt(0)

        para_format.line_spacing = Pt(
            paragraph["font_size"] *
            self.language_config.get_font_multiplier() *
            self.language_config.get_line_spacing_multiplier()
        )
        # TRIAL CONFIG FOR RUBRIK
        para_format.space_before = Pt(0)
        para_format.space_after = Pt(paragraph["font_size"] * 0.5 if index != total - 1 else 0)

        for line, _ in lines:
            for word_info in line:
                run = para.add_run(word_info["text"] + " ")
                run.font.size = Pt(paragraph["font_size"] * self.language_config.get_font_multiplier())
                run.font.name = self.font_name
                r = run._element
                r.rPr.rFonts.set(qn('w:eastAsia'), self.font_name)

                if word_info.get("bold"):
                    run.bold = True
                if word_info.get("italic"):
                    run.italic = True

    @staticmethod
    def _configure_docx_section(docx_doc):
        section = docx_doc.sections[0]

    #     TRIAL CONFIG BASED ON RUBRIK
        # Set A4 size (default, but making it explicit)
        section.page_height = Inches(11.69)  # 297 mm
        section.page_width = Inches(8.27)  # 210 mm

        # Set margins to center a 140mm x 222mm area (5.51in x 8.74in)
        section.top_margin = Inches(1.48)  # (297 - 222)/2 mm
        section.bottom_margin = Inches(1.48)
        section.left_margin = Inches(1.38)  # (210 - 140)/2 mm
        section.right_margin = Inches(1.38)

    def process_pdf(self, input_folder_path: str,output_folder_path: str):
        # 1. Extract text with styling
        pages = self.extract_pages(input_folder_path)

        docx_doc = Document()
        self._configure_docx_section(docx_doc)

        for page in pages:
            page.process_page()
            print(f'[INFO] Page: {page}')

            paragraphs = page.get_paragraphs()

            for i,paragraph in enumerate(paragraphs):
                translated_para =self.translate_preserve_styles(paragraph, self.target_language_key)

                lines = self.layout_paragraph(
                    text=" ".join(translated_para["paragraph"]),
                    bbox=translated_para["bbox"],
                    font_size=paragraph["font_size"],
                    font_path=self.language_config.get_target_font_path(),
                    font_name=self.font_name
                )

                self._write_paragraph_to_docx(docx_doc, lines, paragraph, i, len(paragraphs))


            output_docx_path = os.path.join(output_folder_path, "translated_output.docx")
            docx_doc.save(output_docx_path)


           
            # TODO:
            # 1. REMOVE HEADERS
            # 2. SOME PARAGRAPHS SPLIT ON TO THE NEXT PAGE, AND THOS
            # 2. ADD MARKERS FOR NEW LINE
            # 3. CHECK OUTPUT FOR MULTIPLE PAGES
            # 4. ADD TABS FOR LINES WITHIN THE SAME PARAGRAPHS, AS SOME TEXTS IN THE SAME PARAGRAPHS ARE EITHER CENTERED OR LEFT ALIGNED OR RIGHT ALIGNED,
            # OR HAVE SOME TABS, SO LOOK INTO HOW I CAN MAINTAIN THE FORMAT IN THE TRANSLATED DOCUMENT
            # 5. FOR MULTIPLE PAGES, CHECK LINE SPACING(AND DO WE NEED TO HANDLE LINE SPACING DIFFERENTLY FOR DIFFERENT PAGES)
            # 6. HANDLE HEADERS AND FOOTERS DIFFERENTLY
            # 7. MANAGE FORMAT BETTER

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

