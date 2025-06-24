import os
import re
import sys

import fitz
import torch
from IndicTransToolkit.processor import IndicProcessor
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Pt, Inches
from fitz import Rect
from fontTools.ttLib import TTFont as FontToolsTTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont as ReportlabTTFont
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig

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

        # self.source_font_name = self.load_and_register_font('resource/fonts/Walkman-Chanakya905.ttf')

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

    def initialize_tokenizer(self, ckpt_dir):
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
    def clean_text(text):
        text = re.sub(r'(\d+)([a-z]+)', r'\1 \2', text)
        text = re.sub(r'(\d+)(st|nd|rd|th)', r'\1\2', text)
        cleaned_parts = []
        dot_counts = []

        for segment in re.split(r'(\.{2,})', text):
            if segment.startswith('.') and all(c == '.' for c in segment):
                dot_counts.append(len(segment))
                cleaned_parts.append("[000]")
            else:
                cleaned_parts.append(segment)

        return {"text": "".join(cleaned_parts), "dot_counts": dot_counts}

    @staticmethod
    def restore_dots(translated_text, dot_counts):
        parts = translated_text.split("[000]")
        restored = []

        for i, part in enumerate(parts):
            restored.append(part)
            if i < len(dot_counts):
                restored.append("." * dot_counts[i])

        return "".join(restored)

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

    def extract_spans(self, pdf_path):
        doc = fitz.open(pdf_path)
        spans = []

        for page in doc:
            blocks = page.get_text("dict", sort=True)["blocks"]
            for block in blocks:
                for line in block["lines"]:
                    for span in line["spans"]:
                        font = span["font"]
                        text = span["text"].strip().lower()

                        if 'bold' in font.lower():
                            text = f"[3001]{text}[3002]"
                        elif 'italic' in font.lower():
                            text = f"[3003]{text}[3004]"

                        bbox=fitz.Rect(span['bbox'])

                        if text:
                            spans.append({
                                "text": text,
                                "font": span["font"],
                                "size": span["size"],
                                "page_num": page.number+1,
                                "origin": span["origin"],
                                "bbox": bbox,
                                "line_bbox": fitz.Rect(span['bbox'])
                            })

        return spans

    def get_paragraph_bbox(self , paragraph):
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

        print(f"=================translated {translated}")

        return {
            "paragraph": translated,
            "page_num": paragraph["page_num"],
            "bbox": paragraph_bbox
        }

    def group_into_lines(self, spans):
        """Group text spans into lines based on shared Y position and page number.
        Adds space only when font size increases significantly within the same line.
        """
        lines = []
        current_line = None
        prev_span = None

        def should_insert_space(prev_span, curr_span):
            if not prev_span:
                return False

            x_gap = curr_span["bbox"].x0 - prev_span["bbox"].x1

            # Font size must increase significantly to trigger space
            font_increased = (curr_span["size"] - prev_span["size"]) > 2.0

            same_line = int(prev_span["origin"][1]) == int(curr_span["origin"][1])

            return (x_gap > 1.0) or (font_increased and same_line)

        for span in spans:
            if not span.get("text"):
                continue  # skip empty

            is_new_line = (
                    current_line is None or
                    span["page_num"] != current_line["page_num"] or
                    int(span["origin"][1]) != int(current_line["origin"][1])
            )

            if is_new_line:
                if current_line:
                    lines.append(current_line)
                current_line = {
                    "text": span["text"],
                    "page_num": span["page_num"],
                    "line_bbox": span["line_bbox"],
                    "bbox": fitz.Rect(span["bbox"]),
                    "origin": span["origin"],
                    "size": span["size"]
                }
            else:
                if should_insert_space(prev_span, span):
                    current_line["text"] += " " + span["text"]
                else:
                    current_line["text"] += span["text"]

                # Expand bounding box
                curr_bbox = span["bbox"]
                x0 = min(current_line["bbox"].x0, curr_bbox.x0)
                y0 = min(current_line["bbox"].y0, curr_bbox.y0)
                x1 = max(current_line["bbox"].x1, curr_bbox.x1)
                y1 = max(current_line["bbox"].y1, curr_bbox.y1)
                current_line["bbox"] = fitz.Rect(x0, y0, x1, y1)

                # Update origin
                x = min(current_line["origin"][0], span["origin"][0])
                y = max(current_line["origin"][1], span["origin"][1])
                current_line["origin"] = (x, y)

                # Track highest size so far for reference (optional)
                current_line["size"] = max(current_line["size"], span["size"])

            prev_span = span

        if current_line:
            lines.append(current_line)

        return lines


    def group_into_paragraphs(self, lines):
        """Group text elements into paragraphs based on vertical proximity"""
        paragraphs = []
        current_para = []

        for i, line in enumerate(lines):
            if i > 0 and (line['page_num']!=lines[i-1]['page_num'] or  (int(line['origin'][1])-int(lines[i-1]['origin'][1])>11)):
                if current_para:
                    paragraphs.append({
                        "elements": current_para,
                        "page_num": current_para[0]["page_num"],
                        "original_y": current_para[0]["bbox"].y0,
                        "original_x": current_para[0]["bbox"].x0,
                        "font_size": current_para[0]["size"]
                    })
                    current_para = []
            current_para.append(line)

        if current_para:
            paragraphs.append({
                "elements": current_para,
                "page_num": current_para[0]["page_num"],
                "original_y": current_para[0]["bbox"].y0,
                "original_x": current_para[0]["bbox"].x0,
                "font_size": current_para[0]["size"]
            })

        return paragraphs

    def parse_styled_spans(self,text):
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
                spans.append({"text": buffer, **style})
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

    def layout_paragraph(self,text, bbox, font_size=12, font_name="Times-Roman", font_path=None):
        font = fitz.Font(fontname=font_name, fontfile=font_path)
        styled_spans = self.parse_styled_spans(text)

        print(f'=============styled_spans {styled_spans}')

        max_width = bbox.x1 - bbox.x0
        line_height = font_size * 1.2
        current_y = bbox.y0
        lines = []
        current_line = []

        for span in styled_spans:
            span_words = span["text"].split()
            for word in span_words:
                span_text = word
                test_line = " ".join([s["text"] for s in current_line] + [span_text])
                text_width = font.text_length(test_line, fontsize=font_size)

                if text_width <= max_width:
                    current_line.append({
                        "text": span_text,
                        "bold": span["bold"],
                        "italic": span["italic"]
                    })
                else:
                    if current_line:
                        lines.append((current_line, current_y))
                        current_y += line_height
                    current_line = [{
                        "text": span_text,
                        "bold": span["bold"],
                        "italic": span["italic"]
                    }]

        if current_line:
            lines.append((current_line, current_y))

        total_height = len(lines) * line_height
        adjusted_bbox = fitz.Rect(bbox.x0, bbox.y0, bbox.x1, bbox.y0 + total_height)

        return lines, adjusted_bbox

    def process_pdf(self, input_folder_path: str,output_folder_path: str):
        # 1. Extract text with styling
        spans = self.extract_spans(input_folder_path)

        # print(f"==============spans {spans}")

        sorted_spans = sorted(
            spans,
            key=lambda x: (x["page_num"], x["origin"][1])
        )

        # print(f'==================sorted spans {sorted_spans}')

        # Group spans by lines
        lines = self.group_into_lines(sorted_spans)

        # print(f"==============lines {lines}")

        # 2. Group into paragraphs
        paragraphs = self.group_into_paragraphs(lines)

        # paragraphs = self.add_style_flag(paragraphs)
        #
        print(f'===============paragraphs {paragraphs}')
        #
        #

        docx_doc = Document()
        section = docx_doc.sections[0]
        section.page_height = Inches(11.69)  # A4 height
        section.page_width = Inches(8.27)  # A4 width

        # Adjust margins to maximize usable area
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(0.5)
        section.left_margin = Inches(1.5)
        section.right_margin = Inches(1.5)

        for i,paragraph in enumerate(paragraphs):
            translated_para =self.translate_preserve_styles(paragraph, self.target_language_key)

            print(f"===================translated_para {translated_para}")
        #
            lines, adjusted_bbox = self.layout_paragraph(
                text=" ".join(translated_para["paragraph"]),
                bbox=translated_para["bbox"],
                font_size=paragraph["font_size"],
                font_path=self.language_config.get_target_font_path(),
                font_name=self.font_name
            )
        # #
            print(f'==========================line {lines}')

            para = docx_doc.add_paragraph()
            para_format = para.paragraph_format

            if len(lines) > 1:
                para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                para_format.first_line_indent = Pt(paragraph["font_size"] * self.language_config.get_font_multiplier() * 2)
            else:
                para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                para_format.first_line_indent = Pt(0)

            para_format.line_spacing = Pt(paragraph["font_size"] * self.language_config.get_font_multiplier() * self.language_config.get_line_spacing_multiplier())  # or just Pt(14) for fixed size
            para_format.space_before = Pt(0)
            if i!=len(paragraphs)-1:
                para_format.space_after = Pt(paragraph["font_size"]*0.5)
            else:
                para_format.space_after = Pt(0)


           
            # TODO:
            # 1. REMOVE HEADERS AND FOOTERS(FIRST WORK ON MAINTAING THE FORMAT OF THE MAIN CONTENT AND THEN WORK ON THE HEADERS AND FOOTERS)
            # 2. ADD MARKERS FOR NEW LINE
            # 3. CHECK OUTPUT FOR MULTIPLE PAGES
            # 4. ADD TABS FOR LINES WITHIN THE SAME PARAGRAPHS, AS SOME TEXTS IN THE SAME PARAGRAPHS ARE EITHER CENTERED OR LEFT ALIGNED OR RIGHT ALIGNED,
            # OR HAVE SOME TABS, SO LOOK INTO HOW I CAN MAINTAIN THE FORMAT IN THE TRANSLATED DOCUMENT
            # 5. FOR MULTIPLE PAGES, CHECK LINE SPACING(AND DO WE NEED TO HANDLE LINE SPACING DIFFERENTLY FOR DIFFERENT PAGES)
            # 6. HANDLE HEADERS AND FOOTERS DIFFERENTLY
            # 7. MANAGE FORMAT BETTER
            

            for line, _ in lines:
                run = None
                for word_info in line:
                    run = para.add_run(word_info["text"] + " ")
                    run.font.size = Pt(paragraph["font_size"]*self.language_config.get_font_multiplier())
                    run.font.name = self.font_name

                    # Ensure custom font is applied (especially for non-default fonts)
                    r = run._element
                    r.rPr.rFonts.set(qn('w:eastAsia'), self.font_name)

                    if word_info.get("bold"):
                        run.bold = True
                    if word_info.get("italic"):
                        run.italic = True

        output_docx_path = os.path.join(output_folder_path, "translated_output.docx")
        docx_doc.save(output_docx_path)

