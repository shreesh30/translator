import re
import sys
import os
import logging

import fitz
import torch
from IndicTransToolkit.processor import IndicProcessor
from PIL import ImageFont
from docx import Document
from docx.shared import Inches
from fontTools.ttLib import TTFont as FontToolsTTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont as ReportlabTTFont
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig

from src.model.footer import Footer
from src.model.language_config import LanguageConfig
from src.model.line import Line
from src.service.document_builder import DocumentBuilder
from src.service.document_processor import DocumentProcessor
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
        self.logger = logging.getLogger(__name__)

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
        self.language_config.set_target_font_name(self.font_name)

    def load_and_register_font(self ,font_path):
        try:
            tt = FontToolsTTFont(font_path)
            font_name = tt['name'].getDebugName(1).split('-')[0]
            tt.close()
            self.logger.info(f"Detected font name: {font_name}")
        except Exception as e:
            raise RuntimeError(f"Font metadata read failed: {str(e)}")

        try:
            pdfmetrics.registerFont(ReportlabTTFont(font_name, font_path))
            self.logger.info(f"Successfully registered font: {font_name}")
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
        self.logger.info(f"Translating single text to {tgt_lang}...")
        self.logger.info(f"Translating the text: {text}")

        try:
            processed = self.processor.preprocess_batch([text], src_lang=self.source_language_key, tgt_lang=tgt_lang)
            self.logger.debug(f"Preprocessed: {processed}")

            inputs = self.tokenizer(processed, truncation=True, padding=True, return_tensors="pt").to(
                self.DEVICE)
            self.logger.debug(f"Tokenized Inputs: {inputs}")

            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    min_length=0,
                    max_length=1024,
                    num_beams=6,
                    length_penalty=1.0,
                    early_stopping=True,
                    repetition_penalty=1.1,
                    do_sample=True,
                    temperature = 0.7,
                    no_repeat_ngram_size=3
                )

                self.logger.debug(f"Generated Token IDs: {output}")

            decoded = self.tokenizer.batch_decode(
                    output.detach().cpu().tolist(),
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )

            self.logger.debug(f"Decoded Test: {decoded}")

            final = self.processor.postprocess_batch(decoded, lang=tgt_lang)
            self.logger.debug(f"Postprocessed Translations: {final}")
            self.logger.debug(f"Translated: {final[0]}")

            # Apply replacements
            for old, new in Utils.TAGS.items():
                final[0] = final[0].replace(old, new)

            return final[0]

        except Exception as e:
            self.logger.error(f"Translation failed: {e}")
            return ""


    def translate_preserve_styles(self,paragraph, document_processor):
        # 1. Apply invisible style markers
        text = " ".join(line.get_text() for line in paragraph.get_lines())

        translated_text = self.translate_text(text, self.target_language_key)

        self.logger.info(f'Translated Text: {translated_text}')

        lines = self.split_into_lines(translated_text, paragraph, document_processor)

        new_lines=[]
        for line in lines:
            new_line = self.convert_tags_to_angle_brackets(line)
            line_bbox = self.get_line_bbox(new_line, paragraph.get_font_size())
            new_lines.append(Line(page_number=paragraph.get_page_number(),
                                  text=new_line,
                                  line_bbox= line_bbox,
                                  font_size=paragraph.get_font_size()))

        paragraph.set_lines(new_lines)

        if paragraph.get_sub_paragraphs():
            sub_para = []
            for para in paragraph.get_sub_paragraphs():
                sub_para.append(self.translate_preserve_styles(para, document_processor))
            paragraph.set_sub_paragraphs(sub_para)

        if paragraph.get_footer():
            translated_footer=[]
            for footer in paragraph.get_footer():
                translated_text = self.translate_text(footer.get_text(), self.target_language_key)
                translated_footer.append(Footer(text= translated_text, font_size=footer.get_font_size()))
            paragraph.set_footers(translated_footer)

        if paragraph.get_chapter():
            translated_text = self.translate_text(paragraph.get_chapter(), self.target_language_key)
            paragraph.set_chapter(translated_text)

        if paragraph.get_volume():
            translated_text = self.translate_text(paragraph.get_volume(), self.target_language_key)
            paragraph.set_volume(translated_text)

        return paragraph

    def _add_chapter(self, paragraphs, drawings):
        pass

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

    @staticmethod
    def is_tag(word: str) -> bool:
        return re.fullmatch(r'</?\w+>', word.strip()) is not None

    def split_into_lines(self, para_text, paragraph, document_processor):
        """Splits text into lines using accurate point/inch measurements"""
        font_size = paragraph.get_font_size()

        # 1. Get font metrics in points
        font_size_pt = font_size * self.language_config.get_font_size_multiplier()

        font = ImageFont.truetype(self.language_config.get_target_font_path(), size=font_size_pt)

        # 2. Calculate max width IN POINTS (1 inch = 72 points)
        max_width_pt = self.USABLE_PAGE_WIDTH * 72

        # 3. Language-aware splitting
        return self._split_words(para_text, font, max_width_pt, paragraph, document_processor)

    def _split_words(self ,text, font, max_width_pt, paragraph, document_processor):
        """Splits space-separated languages using accurate width measurements."""
        words = text.split()
        lines = []
        current_line = []
        current_width = 0

        space_width = font.getlength(" ")
        tab_width = 4 * space_width

        para_indent = (
            tab_width if int(paragraph.get_para_bbox().x0) == document_processor.get_paragraph_start()
            else 0
        )

        applied_indent = False

        for i, word in enumerate(words):
            if self.is_tag(word):
                current_line.append(word)
                continue

            word_width = font.getlength(word)

            projected_width = current_width + space_width + word_width

            if not applied_indent:
                projected_width = word_width + para_indent
                applied_indent = True

            if projected_width > max_width_pt:
                lines.append(" ".join(current_line))
                current_line = [word]
                current_width = word_width
            else:
                current_line.append(word)
                current_width = projected_width

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

    def process_pdf(self, input_folder_path: str,output_folder_path: str):
        docx_doc = Document()

        document_processor = DocumentProcessor(input_folder_path, self.language_config)
        document_processor.process_document()

        paragraphs = document_processor.get_paragraphs()
        self.logger.info(f'Paragraphs: {paragraphs}')

        document_builder = DocumentBuilder(document=docx_doc ,language_config= self.language_config, document_processor=document_processor)

        translated_paragraphs = []
        for paragraph in paragraphs:
            translated_para = self.translate_preserve_styles(paragraph, document_processor)
            self.logger.info(f'Translated Paragraph: {translated_para}')
            translated_paragraphs.append(translated_para)

        document_builder.build_document(paragraphs)

        self.logger.info(f'Total Sections: {len(docx_doc.sections)}')

        file_name = os.path.splitext(os.path.basename(input_folder_path))[0]
        output_docx_path = os.path.join(output_folder_path,"{}.docx".format(file_name + '_' + self.language_config.get_target_language()))
        docx_doc.save(output_docx_path)

            # TODO:
            # 2. HANDLE TABLES
            # 3. ADD TRANSLATED HEADERS TO THE TOP OF THE PAGE
            # 4. HANDLE HEADERS AND FOOTERS DIFFERENTLY
            # 5. MANAGE FORMAT BETTER
            # 6. FOR THE CONTENTS TABLE, JUST TRANSLATE THE CONTENTS, LEAVE THE PAGE NUMBERS BLANK

        # ISSUES:
        # 1. EXTRA SPACE IS GETTING ADDED IN EVERY PAGE BETWEEN THE FOOTER AND THE LAST LINE
        # 2. DIFFERENT PAGES REQUIRE DIFFERENT LINE SPACING, SO MAINTAIN THAT


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



