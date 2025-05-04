import re
import sys
import torch

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches
from fontTools.ttLib import TTFont as FontToolsTTFont
from pdf2docx import Converter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont as ReportlabTTFont
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig

from IndicTransToolkit.IndicTransToolkit import IndicProcessor


class PDFTranslator:
    BATCH_SIZE = 10
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    FONT_PATH = "/Users/shreesharya/Documents/Development/Translator/resource/fonts/DVOTSurekh_N_Ship.ttf"
    TMP_DOCX = "/Users/shreesharya/Documents/Development/Translator/resource/tmp/docx/temp.intermediate.docx"
    OUTPUT_DOCX = "/Users/shreesharya/Documents/Development/Translator/resource/output/translated.docx"
    SRC_LANG = "eng_Latn"
    CKPT_DIR = 'ai4bharat/indictrans2-en-indic-1B'

    def __init__(self, quantization=None):
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')

        self.quantization = quantization
        self.tokenizer, self.model = self.initialize_model_and_tokenizer(self.CKPT_DIR)
        self.processor = IndicProcessor(inference=True)
        self.font_name = self.load_and_register_font(self.FONT_PATH)

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
                inputs = self.tokenizer(processed, truncation=True, padding="longest", return_tensors="pt").to(self.DEVICE)
                print(f"[DEBUG] Tokenized Inputs: {inputs}")

                with torch.no_grad():
                    output = self.model.generate(
                        **inputs,
                        min_length=0,
                        max_length=512,
                        num_beams=4,
                        early_stopping=True
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
            except Exception as e:
                print(f"[ERROR] Batch failed: {e}")
                translations.extend([""] * len(batch))

            del inputs
            torch.cuda.empty_cache()

        for elem, trans in zip(elements, translations):
            elem['content'] = trans

    def process_pdf(self, input_pdf: str, tgt_lang: str):
        Converter(input_pdf).convert(self.TMP_DOCX,start=0, end=None, layout_analysis=True, keep_blank_lines=True, table_detection=True)

        doc = Document(self.TMP_DOCX)
        for section in doc.sections:
            section.top_margin = Inches(0)
            section.bottom_margin = Inches(0)

        table_idx = para_idx = 0
        for block in doc.element.body:
            elements = []

            if block.tag.endswith('tbl') and table_idx < len(doc.tables):
                table = doc.tables[table_idx]
                table_idx += 1
                for row in table.rows:
                    for cell in row.cells:
                        for para in cell.paragraphs:
                            para.alignment = WD_ALIGN_PARAGRAPH.DISTRIBUTE
                            for run in para.runs:
                                self._process_run(run, elements)

            elif block.tag.endswith('p') and para_idx < len(doc.paragraphs):
                paragraph = doc.paragraphs[para_idx]
                para_idx += 1
                for run in paragraph.runs:
                    self._process_run(run, elements)

            if elements:
                self.batch_translate(elements, tgt_lang)
                for elem in elements:
                    self._update_run(elem)

        doc.save(self.OUTPUT_DOCX)

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