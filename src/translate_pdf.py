import re
import re
import sys
import time

import ocrmypdf
import torch
from PIL import Image

from docx import Document
from docx.enum.text import WD_LINE_SPACING, WD_TAB_ALIGNMENT
from docx.shared import Pt, RGBColor, Inches, Emu
from fontTools.ttLib import TTFont as FontToolsTTFont
from pdf2docx import Converter
import json
import io
import pytesseract
from bs4 import BeautifulSoup
import fitz
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont as ReportlabTTFont
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig
from IndicTransToolkit.processor import IndicProcessor
from adobe.pdfservices.operation.auth.service_principal_credentials import ServicePrincipalCredentials
from adobe.pdfservices.operation.exception.exceptions import ServiceApiException, ServiceUsageException, SdkException
from adobe.pdfservices.operation.io.cloud_asset import CloudAsset
from adobe.pdfservices.operation.io.stream_asset import StreamAsset
from adobe.pdfservices.operation.pdf_services import PDFServices
from adobe.pdfservices.operation.pdf_services_media_type import PDFServicesMediaType
from adobe.pdfservices.operation.pdfjobs.jobs.export_pdf_job import ExportPDFJob
from adobe.pdfservices.operation.pdfjobs.params.export_pdf.export_pdf_params import ExportPDFParams
from adobe.pdfservices.operation.pdfjobs.params.export_pdf.export_pdf_target_format import ExportPDFTargetFormat
from adobe.pdfservices.operation.pdfjobs.result.export_pdf_result import ExportPDFResult


# from IndicTransToolkit.IndicTransToolkit import IndicProcessor


class PDFTranslator:
    BATCH_SIZE = 10
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # HINDI
    FONT_PATH = "resource/fonts/DVOTSurekh_N_Ship.ttf"
    # ODIA
    # FONT_PATH = "resource/fonts/SakalBharati_N_Ship.ttf"
    # BENGALI
    # FONT_PATH = "resource/fonts/bnotdurga_n_ship.ttf"
    # TMP_DOCX = "/Users/shreesharya/Documents/Development/Translator/resource/tmp/docx/temp.intermediate.docx"
    TMP_DOCX = "resource/tmp/docx/temp.intermediate.docx"
    # OUTPUT_DOCX = "/Users/shreesharya/Documents/Development/Translator/resource/output/translated.docx"
    OUTPUT_DOCX = "resource/output/translated.docx"
    SRC_LANG = "eng_Latn"
    CKPT_DIR = 'ai4bharat/indictrans2-en-indic-1B'
    BOLD_TAG='[111]'

    def __init__(self, quantization=None):
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')

        self.quantization = quantization
        self.tokenizer, self.model = self.initialize_model_and_tokenizer(self.CKPT_DIR)
        self.processor = IndicProcessor(inference=True)
        self.font_name = self.load_and_register_font(self.FONT_PATH)
        self.tab_stops_created = set()

        # self.model = lp.Detectron2LayoutModel(
        #     config_path="lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config.yaml",
        #     model_path="lp://PubLayNet/faster_rcnn_R_50_FPN_3x/model_final.pth",
        #     label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
        # )

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

    def initialize_model_and_tokenizer(self, ckpt_dir):
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

        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            ckpt_dir,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            quantization_config=qconfig,
        )

        if qconfig is None:
            model = model.to(self.DEVICE)
            if self.DEVICE == "cuda":
                model.half()

        model.eval()
        return tokenizer, model

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
                processed = self.processor.preprocess_batch(batch, src_lang=self.SRC_LANG, tgt_lang=tgt_lang)
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

                with self.tokenizer.as_target_tokenizer():
                    decoded = self.tokenizer.batch_decode(
                        output.detach().cpu().tolist(),
                        skip_special_tokens=True,
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

    @staticmethod
    def normalize_font_size_in_docx(input_path, output_path, target_size_pt=12):
        doc = Document(input_path)

        for para in doc.paragraphs:
            new_runs = []
            current_word = ''
            for run in para.runs:
                # Accumulate text and remember styling
                if re.match(r'\s', run.text):
                    # Write the previous word as a single run
                    if current_word:
                        new_run = para.add_run(current_word)
                        new_run.font.size = Pt(target_size_pt)
                        current_word = ''
                    new_run = para.add_run(run.text)
                    new_run.font.size = Pt(target_size_pt)
                else:
                    current_word += run.text

            # Handle trailing word
            if current_word:
                new_run = para.add_run(current_word)
                new_run.font.size = Pt(target_size_pt)

            # Remove old runs (by clearing text)
            for run in para.runs[:-len(para.runs)]:
                run.text = ""

        doc.save(output_path)

    # def parse_style(self,style_str):
    #     styles = {}
    #     if not style_str:
    #         return styles
    #     for part in style_str.split(';'):
    #         if ':' in part:
    #             key, val = part.split(':', 1)
    #             styles[key.strip()] = val.strip()
    #     return styles
    #
    # def hex_to_rgb(self,hex_color):
    #     hex_color = hex_color.strip("#")
    #     return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    #
    # def detect_rotation(self,image):
    #     osd = pytesseract.image_to_osd(image)
    #     return int(re.search(r'Rotate: (\d+)', osd).group(1))
    #
    # def extract_text_by_region(self,page_image):
    #     width, height = page_image.size
    #
    #     # Define header and body split (top 15% is header)
    #     header_box = (0, 0, width, int(height * 0.15))
    #     body_box = (0, int(height * 0.15), width, height)
    #
    #     header_img = page_image.crop(header_box)
    #     body_img = page_image.crop(body_box)
    #
    #     # Detect and correct rotation of body
    #     body_rotation = self.detect_rotation(body_img)
    #     if body_rotation != 0:
    #         body_img = body_img.rotate(360 - body_rotation, expand=True)
    #
    #     # OCR
    #     header_text = pytesseract.image_to_data(header_img, output_type=pytesseract.Output.DICT)
    #     body_text = pytesseract.image_to_data(body_img, output_type=pytesseract.Output.DICT)
    #
    #     return header_text, body_text

    # def add_text_from_data(self,data,doc, top_offset=0):
    #     n_boxes = len(data['text'])
    #     elements = []
    #     for i in range(n_boxes):
    #         if int(data['conf'][i]) > 30 and data['text'][i].strip():
    #             text = data['text'][i]
    #             top = data['top'][i] + top_offset
    #             left = data['left'][i]
    #             height = data['height'][i]
    #             elements.append((top, left, height, text))
    #
    #     elements.sort(key=lambda e: (e[0], e[1]))
    #
    #     for top, left, height, text in elements:
    #         para = doc.add_paragraph()
    #         run = para.add_run(text)
    #         run.font.size = Pt(height * 0.75 / 1.33)
    #         para.paragraph_format.line_spacing_rule = WD_LINE_SPACING.EXACTLY
    #         para.paragraph_format.line_spacing = Pt(height * 0.9)
    #         para.paragraph_format.left_indent = Inches(left / 72)

    # def convert_pdf_to_docx_test(self,pdf_path, output_path):
    #     doc = Document()
    #     section = doc.sections[0]
    #     section.page_width = Inches(595 / 72)  # A4 width in inches
    #     section.page_height = Inches(842 / 72)  # A4 height in inches
    #     section.top_margin = section.bottom_margin = section.left_margin = section.right_margin = Inches(0)
    #
    #     pdf = fitz.open(pdf_path)
    #
    #     for page_num, page in enumerate(pdf):
    #         blocks = page.get_text("dict")["blocks"]
    #
    #         print(f"===============blocks {blocks}")
    #
    #         for block in blocks:
    #             print(f"===============++++++++++++++++++block {block}")
    #             if block['type'] == 0:  # text block
    #                 for line in block["lines"]:
    #                     for span in line["spans"]:
    #                         text = span["text"].strip()
    #                         if not text:
    #                             continue
    #
    #                         font_size = span["size"]
    #                         color = span["color"]
    #                         font = span["font"]
    #                         bbox = span["bbox"]
    #                         top_pt = bbox[1]
    #                         left_pt = bbox[0]
    #
    #                         paragraph = doc.add_paragraph()
    #                         run = paragraph.add_run(text)
    #                         run.font.size = Pt(font_size)
    #                         run.font.name = font
    #                         run.font.color.rgb = RGBColor((color >> 16) & 255, (color >> 8) & 255, color & 255)
    #
    #                         # Simulate positioning with spacing (limitation: python-docx has no absolute position)
    #                         paragraph.paragraph_format.left_indent = Inches(left_pt / 72)
    #                         paragraph.paragraph_format.space_before = Pt(top_pt)
    #
    #             elif block['type'] == 1:  # image block
    #                 img_index = block.get("image")
    #                 xref = block["image"]
    #                 base_image = pdf.extract_image(xref)
    #                 image_bytes = base_image["image"]
    #                 img = Image.open(io.BytesIO(image_bytes))
    #                 img_path = f"page_{page_num}_img.png"
    #                 img.save(img_path)
    #
    #                 doc.add_picture(img_path, width=Inches(6))  # Approximate width
    #
    #     doc.save(output_path)



    def _extract_and_sort_spans_from_page(self,page):
        """
        Extracts all text spans from a PDF page and sorts them
        by their top Y-coordinate, then by their left X-coordinate.
        """
        all_spans_on_page = []
        for block in page.get_text("dict")["blocks"]:
            if block["type"] == 0 and block.get("lines"):  # Only process text blocks (type 0)
                for line in block["lines"]:
                    all_spans_on_page.extend(line["spans"])

        # Sort all spans initially by their top Y-coordinate, then by their left X-coordinate.
        all_spans_on_page.sort(key=lambda s: (s["bbox"][1], s["bbox"][0]))
        return all_spans_on_page

    def _group_spans_into_logical_lines(self,all_spans_on_page, line_vertical_tolerance):
        """
        Groups individual spans into logical lines based on vertical proximity.
        """
        logical_lines = []
        current_logical_line_spans = []
        current_line_y_reference = None  # Reference Y-coordinate for the current logical line

        for span in all_spans_on_page:
            span_y_top = span["bbox"][1]

            if not current_logical_line_spans:
                current_logical_line_spans.append(span)
                current_line_y_reference = span_y_top
            elif abs(span_y_top - current_line_y_reference) < line_vertical_tolerance:
                current_logical_line_spans.append(span)
            else:
                logical_lines.append(current_logical_line_spans)
                current_logical_line_spans = [span]
                current_line_y_reference = span_y_top

        if current_logical_line_spans:
            logical_lines.append(current_logical_line_spans)
        return logical_lines

    def _process_logical_line_into_docx_paragraph(self,paragraph, logical_line_spans, prev_docx_paragraph_bottom_y, doc_left_margin_inches):
        """
        Processes a single logical line of spans and adds them as runs to a DOCX paragraph,
        handling vertical and horizontal spacing, and text formatting.
        """
        # Conversion factor for DOCX measurements
        POINTS_TO_EMU = 12700
        # CRUCIAL STEP: Re-sort spans within this logical line by their X-coordinate (left edge).
        logical_line_spans.sort(key=lambda s: s["bbox"][0])

        # Reset default paragraph spacing to ensure precise control
        paragraph.paragraph_format.space_before = Pt(0)
        paragraph.paragraph_format.space_after = Pt(0)
        paragraph.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE

        first_span_y_top = logical_line_spans[0]["bbox"][1]
        first_span_x_left = logical_line_spans[0]["bbox"][0]

        # Calculate vertical spacing from the previous DOCX paragraph
        if prev_docx_paragraph_bottom_y is not None:
            vertical_gap_points = first_span_y_top - prev_docx_paragraph_bottom_y
            if vertical_gap_points > 0.1:  # Only add space if there's a significant positive gap
                paragraph.paragraph_format.space_before = Pt(vertical_gap_points)

        # Calculate left indentation for the new DOCX paragraph
        docx_left_margin_points = doc_left_margin_inches * 72
        desired_indent_points = first_span_x_left - docx_left_margin_points
        if desired_indent_points < 0:
            desired_indent_points = 0  # Ensure non-negative indent
        paragraph.paragraph_format.left_indent = Emu(desired_indent_points * POINTS_TO_EMU)

        prev_span_end_x = None  # Tracks the right X-coordinate of the last added span for horizontal spacing
        max_span_y_bottom = 0  # Tracks the maximum (lowest) Y-coordinate among spans in the current logical line

        for span_index, span in enumerate(logical_line_spans):
            text = span["text"]
            span_bbox_x1 = span["bbox"][0]  # Start X of span's bounding box
            span_bbox_x2 = span["bbox"][2]  # End X of span's bounding box

            # Update max_span_y_bottom for the current logical line
            max_span_y_bottom = max(max_span_y_bottom, span["bbox"][3])

            # Add horizontal spaces between spans if there's a significant gap
            if prev_span_end_x is not None:
                horizontal_gap = span_bbox_x1 - prev_span_end_x
                if horizontal_gap > 0.5:  # Small threshold to ignore minor, unintentional gaps
                    estimated_space_width = span.get("size", 10) * 0.4  # Using 0.4 as per your last code
                    if estimated_space_width < 1: estimated_space_width = 5
                    num_spaces = round(horizontal_gap / estimated_space_width)
                    if num_spaces > 0:
                        paragraph.add_run(" " * num_spaces)

            # Add the actual text run to the paragraph
            run = paragraph.add_run(text)
            run.font.size = Pt(span.get("size", 11))
            run.font.name = span.get("font", "Times New Roman")

            # Set font color
            color = span["color"]
            run.font.color.rgb = RGBColor((color >> 16) & 255, (color >> 8) & 255, color & 255)

            prev_span_end_x = span_bbox_x2  # Update the end X for the next span's spacing calculation

        return max_span_y_bottom

    def convert_pdf_to_docx(self,pdf_path, docx_path, line_vertical_tolerance=3):
        """
        Converts a PDF document to DOCX, using advanced span-based grouping to preserve layout
        and horizontal order more accurately.

        Args:
            pdf_path (str): Path to the input PDF file.
            docx_path (str): Path to save the output DOCX file.
            line_vertical_tolerance (float): Maximum vertical difference (in points)
                                             to consider two spans as being on the same
                                             logical line. A higher value allows more
                                             vertical "drift" within a line.
                                             Adjust this value based on your PDF if needed.
        """
        doc = Document()
        pdf = fitz.open(pdf_path)

        for page_num, page in enumerate(pdf):
            rect = page.rect
            pdf_width_in_points = rect.width
            pdf_height_in_points = rect.height

            if page_num == 0:
                section = doc.sections[0]
            else:
                section = doc.add_section()

            # Set DOCX page dimensions to match PDF
            section.page_width = Inches(pdf_width_in_points / 72)
            section.page_height = Inches(pdf_height_in_points / 72)

            # Set DOCX page margins. Adjust these based on your PDF's visual margins.
            doc_left_margin_inches = 0.3
            doc_top_margin_inches = 0.5
            doc_bottom_margin_inches = 0.1
            doc_right_margin_inches = 0.3

            section.top_margin = Inches(doc_top_margin_inches)
            section.bottom_margin = Inches(doc_bottom_margin_inches)
            section.left_margin = Inches(doc_left_margin_inches)
            section.right_margin = Inches(doc_right_margin_inches)

            # Extract and group spans
            all_spans_on_page = self._extract_and_sort_spans_from_page(page)
            logical_lines = self._group_spans_into_logical_lines(all_spans_on_page, line_vertical_tolerance)

            prev_docx_paragraph_bottom_y = None

            # Process each identified logical line into a DOCX paragraph
            for logical_line_spans in logical_lines:
                paragraph = doc.add_paragraph()
                prev_docx_paragraph_bottom_y = self._process_logical_line_into_docx_paragraph(
                    paragraph, logical_line_spans, prev_docx_paragraph_bottom_y, doc_left_margin_inches
                )

            # Add a page break after each PDF page, except the last one
            if page_num < len(pdf) - 1:
                doc.add_page_break()

        # Save the final DOCX document
        doc.save(docx_path)
        print(f"Saved to: {docx_path}")

    def extract_style_attributes(self,style_str):
        """
        Parse a style string like 'font-size:11.0pt;color:#000000;font-family:Times New Roman'
        into a dict.
        """
        styles = {}
        for item in style_str.split(';'):
            if ':' in item:
                key, val = item.split(':', 1)
                styles[key.strip()] = val.strip()
        return styles


    def process_pdf(self, input_pdf: str, tgt_lang: str):
        start = int(time.time())

        # try:
        #
        # ocrmypdf.ocr(
        #     input_file=input_pdf,
        #     output_file=ocr_path,
        #     optimize=3,
        #     deskew=True,
        #     clean=True,
        #     clean_final=True,
        #     force_ocr=True,
        #     output_type='pdf',
        #     image_dpi=300,
        #     pdf_renderer='sandwich',
        #     language='eng'  # or 'eng+hin' for multiple languages
        # )

        # self.convert(input_pdf,'output.docx')
        # self.convert_pdf_to_docx_test(input_pdf, self.TMP_DOCX)
        self.convert_pdf_to_docx(input_pdf, self.TMP_DOCX)

        # doc = Document()
        #
        # # Set page size to A4 (595pt x 842pt)
        # section = doc.sections[0]
        # section.page_width = Inches(595 / 72)  # 8.26 inches
        # section.page_height = Inches(842 / 72)  # 11.69 inches
        #
        # # for section in doc.sections:
        # section.top_margin = Inches(0)
        # section.bottom_margin = Inches(0)
        #
        # ocr_pdf = fitz.open(input_pdf)
        #
        # idx=0
        # elements = []
        # for page in ocr_pdf:
        #     # print(f'===============page {page}')
        #     # print(f'===============page text {page.get_text("text")}')
        #     # print(f'===============page dict {page.get_text("dict")}')
        #     # print(f'===============page blocks {page.get_text("blocks")}')
        #     # print(f'===============page words {page.get_text("words")}')
        #     print(f'===============page html {page.get_text("html")}')
        #     # print(f"=============page rotation {page.rotation}")
        #     html = page.get_text("html")
        #     soup = BeautifulSoup(html, 'html.parser')
        #
        #     for p in soup.find_all('p'):
        #         style = self.parse_style(p.get('style', ''))
        #         top = float(style.get('top', '0').replace('pt', ''))
        #         left = float(style.get('left', '0').replace('pt', ''))
        #         elements.append((top, left, p))
        #
        #     elements.sort(key=lambda x: (x[0], x[1]))
        #
        #     print(f'========================elements {elements}')
        #
        # if elements:
        #     for top, left, p in elements:
        #         # Extract text and styles
        #         text = p.get_text(strip=True)
        #         span = p.find('span')
        #         span_style = self.parse_style(span.get('style', '')) if span else {}
        #
        #         # print(f"==========text {text}")
        #         # print(f"==========span_style {span_style}")
        #
        #         # Font properties
        #         font_family = span_style.get('font-family', 'Times New Roman').split(',')[0].strip()
        #         font_size = float(re.sub(r'[^\d.]', '', span_style.get('font-size', '12pt')))
        #         color = span_style.get('color', '#000000')
        #         rgb = self.hex_to_rgb(color)
        #
        #         # Bold/Italic
        #         is_bold = p.find('b') is not None
        #         is_italic = p.find('i') is not None
        #
        #         # Create paragraph
        #         paragraph = doc.add_paragraph()
        #
        #         # Apply line height
        #         line_height = float(p.get('style', '').split('line-height:')[-1].split('pt')[0].strip())
        #         if line_height:
        #             paragraph.paragraph_format.line_spacing_rule = WD_LINE_SPACING.EXACTLY
        #         paragraph.paragraph_format.line_spacing = Pt(float(line_height))
        #
        #         # Add text with styles
        #         run = paragraph.add_run(text)
        #         run.font.name = font_family
        #         run.font.size = Pt(font_size)
        #         run.font.color.rgb = RGBColor(*rgb)
        #         run.bold = is_bold
        #         run.italic = is_italic
        #
        # doc.save('output.docx')




            # print(f'===============page term {page.get_text("term")}')

            # blocks = page.get_text("dict")["blocks"]


        #     with open('resource/credentials/pdfservices-api-credentials.json', 'r') as file:
        #         data = json.load(file)
        #
        #     credentials = ServicePrincipalCredentials(
        #         client_id = data['client_credentials']['client_id'],
        #         client_secret=data['client_credentials']['client_secret'])
        #
        #     # Creates a PDF Services instance
        #     pdf_services = PDFServices(credentials=credentials)
        #
        #     file = open(ocr_path, 'rb')
        #     input_stream = file.read()
        #     file.close()
        #
        #     # Creates an asset(s) from source file(s) and upload
        #     # input_asset = pdf_services.upload(input_stream=input_stream, mime_type=PDFServicesMediaType.PDF)
        #
        #     with open(ocr_path, 'rb') as pdf_file:
        #         input_asset = pdf_services.upload(
        #             input_stream=pdf_file,
        #             mime_type=PDFServicesMediaType.PDF
        #         )
        #
        #     # Create parameters for the job
        #     export_pdf_params = ExportPDFParams(target_format=ExportPDFTargetFormat.DOCX)
        #
        #     # Creates a new job instance
        #     export_pdf_job = ExportPDFJob(input_asset=input_asset, export_pdf_params=export_pdf_params)
        #
        #     # Submit the job and gets the job result
        #     location = pdf_services.submit(export_pdf_job)
        #     pdf_services_response = pdf_services.get_job_result(location, ExportPDFResult)
        #
        #     # Get content from the resulting asset(s)
        #     result_asset: CloudAsset = pdf_services_response.get_result().get_asset()
        #     stream_asset: StreamAsset = pdf_services.get_content(result_asset)
        #
        #     output_file_path = self.TMP_DOCX
        #     with open(output_file_path, "wb") as file:
        #         file.write(stream_asset.get_input_stream())
        #
        # except (ServiceApiException, ServiceUsageException, SdkException) as e:
        #     print(f'Exception encountered while executing operation: {e}')

        # Step 1: Convert PDF to DOCX
        # cv = Converter(input_pdf)
        # cv.convert(self.TMP_DOCX, start=0, end=None, debug=True)
        # cv.close()

        # convert(self.TMP_DOCX, 'resource/tmp/docx')
        # doc = Document(self.TMP_DOCX)
        #
        # for paragraph in doc.paragraphs:
        #     if paragraph.text.strip():
        #         for run in paragraph.runs:


        # doc.save(self.TMP_DOCX)

        #
        # table_idx = para_idx = 0
        #
        # for block in doc.element.body:
        #     print(f'==================block.tag {block.tag}')
        #     # print(f'==================doc.paragraphs length {len(doc.paragraphs)}')
        #     if block.tag.endswith('p') and para_idx < len(doc.paragraphs):
        #         paragraph = doc.paragraphs[para_idx]
        #         # print(f'==============para {paragraph.runs}')
        #         if paragraph.paragraph_format.page_break_before:
        #             print("****************Paragraph starts a new page:", paragraph.text)
        #
        #         for run in paragraph.runs:
        #             print(f'================para.runs {run.text}')
        #             # print(f'=================={run._element.xml}')
        #         para_idx += 1


            # for para in doc.paragraphs:
            #     print(f'================para.runs {para.runs}')
            #     for run in para.runs:
            #         print(f'================para.runs {run.text}')

        # convert(self.TMP_DOCX, "resource/output/output.pdf")

        # Step 2: Load the DOCX
        # doc = Document(self.TMP_DOCX)
        #
        # # Remove top/bottom margins
        # for section in doc.sections:
        #     section.top_margin = Inches(0)
        #     section.bottom_margin = Inches(0)
        #
        # table_idx = para_idx = 0
        # #
        # # for block in doc.element.body:
        # #     print(etree.tostring(block, pretty_print=True).decode())
        #
        # for block in doc.element.body:
        #     print(f'==================block.tag {block.tag}')
        #     print(f'==================doc.paragraphs length {len(doc.paragraphs)}, table_idx {table_idx}')
        #     if block.tag.endswith('tbl') and table_idx < len(doc.tables):
        #         table = doc.tables[table_idx]
        #         table_idx += 1
        #         for row in table.rows:
        #             for cell in row.cells:
        #                 for para in cell.paragraphs:
        #                     # TODO: HANDLE TABLE CELLS, DON'T USE THIS METHOD
        #                     self._translate_paragraph_with_style(para, tgt_lang)
        #
        #     elif block.tag.endswith('p') and para_idx < len(doc.paragraphs):
        #         paragraph = doc.paragraphs[para_idx]
        #         para_idx += 1
        #         print(f"=============paragraph {paragraph}")
        #         self._translate_paragraph_with_style(paragraph, tgt_lang)
        #
        # # Step 3: Save the translated DOCX
        # doc.save(self.OUTPUT_DOCX)
        # print(f"========= Finished in {int(time.time()) - start} seconds")



    def _process_run(self, run, elements):
        text = run.text
        if not text or not text.strip() or not self.is_word_only(text):
            return
        cleaned = self.clean_text(text.strip())
        elements.append({
            'content': cleaned['text'].lower() if text.isupper() else cleaned['text'],
            'run': run,
            'dot_counts': cleaned['dot_counts']
        })

    def _update_run(self, elem):
        translated = elem['content']
        if not translated:
            return
        if "[000]" in translated:
            translated = self.restore_dots(translated, elem['dot_counts'])
        run = elem['run']
        run.text = translated
        run.font.name = self.font_name
        run.font.italic = False

    def _translate_paragraph_with_style(self, paragraph, tgt_lang: str):
        # TODO: ADD TAG FOR .... AS WELL
        def preprocess_runs(runs):
            parts = []
            for run in runs:
                text = run.text.lower() if run.text.isupper() else run.text
                print(f"==========text {text}")
                if not text.strip():
                    continue
                parts.append(f"{self.BOLD_TAG}{text}{self.BOLD_TAG}" if run.bold else text)
            return ''.join(parts)

        def apply_style(run, text, bold, font_size):
            run.text = text
            run.font.bold = bold
            run.font.size = font_size
            run.font.name = self.font_name

        runs = paragraph.runs
        tagged_text = preprocess_runs(runs)

        print(f"==============tagged_text {tagged_text}")
        if not tagged_text.strip():
            return

        original_font_size = runs[0].font.size

        try:
            translated_text = self.translate_paragraph(tagged_text, tgt_lang)
            paragraph.clear()

            bold_tag = re.escape(self.BOLD_TAG)
            pattern = f'({bold_tag}.*?{bold_tag})'
            tokens = re.split(pattern, translated_text)

            for token in filter(str.strip, tokens):
                is_bold = token.startswith(self.BOLD_TAG) and token.endswith(self.BOLD_TAG)
                clean_text = token[5:-5] if is_bold else token
                apply_style(paragraph.add_run(), clean_text, is_bold, original_font_size)

            paragraph.paragraph_format.line_spacing_rule = WD_LINE_SPACING.MULTIPLE

        except Exception as e:
            print(f"[ERROR] Paragraph translation failed: {e}")

    def translate_paragraph(self, text: str, tgt_lang: str) -> str:
        elements = [{"content": text}]
        self.batch_translate(elements, tgt_lang)
        return elements[0]["content"]
