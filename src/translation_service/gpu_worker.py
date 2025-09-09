import gzip
import json
import logging
import pickle
import re
from dataclasses import asdict

import fitz
import torch
from IndicTransToolkit.processor import IndicProcessor
from PIL import ImageFont
from dacite import from_dict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig

from src.model.bbox import Bbox
from src.model.footer import Footer
from src.model.language_config import LanguageConfig
from src.model.line import Line
from src.model.paragraph import Paragraph
from src.model.result import Result
from src.model.table import Table
from src.model.task import Task
from src.utils.rabbitmq_consumer import RabbitMQConsumer
from src.utils.rabbitmq_producer import RabbitMQProducer
from src.utils.utils import Utils

logger = logging.getLogger(Utils.TRANSLATION_SERVICE)

class GPUWorker:
    def __init__(self, model_name: str, quantization: str):
        super().__init__()
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.quantization = quantization
        self.processor = None
        self.tags = None
        self.producer = None

    @staticmethod
    def is_tag(word: str) -> bool:
        return re.fullmatch(r'</?\w+>', word.strip()) is not None

    def split_into_lines(self, para_text, paragraph, task):
        """Splits text into lines using accurate point/inch measurements"""
        font_size = paragraph.get_font_size()
        language_config = task.language_config

        # 1. Get font metrics in points
        font_size_pt = font_size * language_config.get_font_size_multiplier()

        font = ImageFont.truetype(language_config.get_target_font_path(), size=font_size_pt)

        # 2. Calculate max width IN POINTS (1 inch = 72 points)
        max_width_pt = Utils.USABLE_PAGE_WIDTH * 72

        # 3. Language-aware splitting
        return self._split_words(para_text, font, max_width_pt, paragraph, task)

    def _split_words(self ,text, font, max_width_pt, paragraph, task):
        """Splits space-separated languages using accurate width measurements."""
        meta_data = task.meta_data
        paragraph_start = meta_data.get_paragraph_start()

        words = text.split()
        lines = []
        current_line = []
        current_width = 0

        space_width = font.getlength(" ")
        tab_width = 4 * space_width

        para_indent = (
            tab_width if int(paragraph.get_para_bbox().x0) == paragraph_start
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

    @staticmethod
    def remove_angle_brackets(text: str) -> str:
        """
        Removes surrounding angle brackets like <...> but keeps inner content.
        """
        # Remove one layer of angle brackets if the entire string is wrapped in them
        cleaned = re.sub(r'^<\s*(.*?)\s*>$', r'\1', text)

        return cleaned.strip()

    @staticmethod
    def remove_consecutive_dots(text):
        cleaned = re.sub(r'\.{3,}', '', text)
        return cleaned

    @staticmethod
    def convert_tags_to_angle_brackets(text: str) -> str:
        # Replace <b>...</b> or <i>...</i> with <...>
        text = re.sub(r'<[^>/]+>(.*?)</[^>]+>', r'<\1>', text)

        return text.strip()

    def get_line_bbox(self, line, font_size, language_config):
        line_copy = str(line)
        line_copy = self.remove_angle_brackets(line_copy)

        font_size_pt = font_size * language_config.get_font_size_multiplier()
        # Load font
        font = ImageFont.truetype(language_config.get_target_font_path(), size=font_size_pt)
        bbox = font.getbbox(line_copy)
        fitz_bbox = fitz.Rect(bbox)
        return Bbox(x0=fitz_bbox.x0,x1=fitz_bbox.x1,y0=fitz_bbox.y0,y1=fitz_bbox.y1)

    def translate_paragraph(self,paragraph : Paragraph, task):
        try:
            language_config = task.language_config
            # 1. Apply invisible style markers
            target_language_key  = language_config.get_target_language_key()
            source_language_key = language_config.get_source_language_key()

            text = " ".join(line.get_text() for line in paragraph.get_lines())

            translated_text = self.translate_text(text, target_language_key, source_language_key)

            logger.info(f'Translated Text: {translated_text}')

            lines = self.split_into_lines(translated_text, paragraph, task)

            new_lines=[]
            for line in lines:
                new_line = self.convert_tags_to_angle_brackets(line)
                line_bbox = self.get_line_bbox(new_line, paragraph.get_font_size(), language_config)
                new_lines.append(Line(page_number=paragraph.get_page_number(),
                                      text=new_line,
                                      line_bbox= line_bbox,
                                      font_size=paragraph.get_font_size()))

            paragraph.set_lines(new_lines)

            if paragraph.get_sub_paragraphs():
                sub_para = []
                for para in paragraph.get_sub_paragraphs():
                    sub_para.append(self.translate_paragraph(para, task))
                paragraph.set_sub_paragraphs(sub_para)

            if paragraph.get_footer():
                translated_footer=[]
                for footer in paragraph.get_footer():
                    footer_text = footer.get_text()
                    pattern = r'(?i)\benglish(?=\s+translation\s+of)'
                    footer_text = re.sub(pattern, language_config.get_target_language(), footer_text)

                    translated_text = self.translate_text(footer_text , target_language_key, source_language_key)
                    translated_footer.append(Footer(text= translated_text, font_size=footer.get_font_size()))
                paragraph.set_footers(translated_footer)

            if paragraph.get_chapter():
                translated_text = self.translate_text(paragraph.get_chapter(), target_language_key, source_language_key)
                paragraph.set_chapter(translated_text)

            if paragraph.get_volume():
                translated_text = self.translate_text(paragraph.get_volume(), target_language_key, source_language_key)
                paragraph.set_volume(translated_text)
        except Exception as e:
            logger.error(f'Error translating paragraph: {paragraph}: {e}', exc_info=True)

        return paragraph

    def translate_table(self, table: Table, language_config: LanguageConfig):
        logger.info(f'Translate Table: {table}')
        try:
            target_language_key = language_config.get_target_language_key()
            source_language_key = language_config.get_source_language_key()

            if table.get_title().get_text():
                title_line = table.get_title()
                text = title_line.get_text().lower()

                translated_text = self.translate_text(text, target_language_key, source_language_key)
                logger.info(f'Translated Text: {translated_text}')
                title_line.set_text(translated_text)

            if table.get_sub_title().get_text():
                sub_title_line = table.get_sub_title()
                text = sub_title_line.get_text().lower()

                translated_text = self.translate_text(text, target_language_key, source_language_key)
                logger.info(f'Translated Text: {translated_text}')
                sub_title_line.set_text(translated_text)

            if table.get_columns():
                for column in table.get_columns():
                    for row in column:
                        text = row.get_text().lower()
                        cleaned_text = self.remove_consecutive_dots(text)

                        translated_text = self.translate_text(cleaned_text, target_language_key, source_language_key)
                        logger.info(f'Translated Text: {translated_text}')
                        row.set_text(translated_text)
        except Exception as e:
            logger.error(f'Error translating table:{table}: {e}', exc_info=True)

        return table

    def translate_text(self, text: str, tgt_lang: str, src_lang:str) -> str:
        logger.info(f"Translating single text to {tgt_lang}...")
        logger.info(f"Translating the text: {text}")

        try:
            processed = self.processor.preprocess_batch([text], src_lang=src_lang, tgt_lang=tgt_lang)
            logger.info(f"Preprocessed: {processed}")

            inputs = self.tokenizer(processed, truncation=True, padding=True, return_tensors="pt").to("cuda")

            logger.info(f"Tokenized Inputs: {inputs}")
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
                        no_repeat_ngram_size=3,
                        use_cache = False
                    )

                logger.info(f"Generated Token IDs: {output}")

            decoded = self.tokenizer.batch_decode(
                    output.detach().cpu().tolist(),
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )

            logger.info(f"Decoded Test: {decoded}")

            final = self.processor.postprocess_batch(decoded, lang=tgt_lang)
            logger.info(f"Post processed Translations: {final}")
            logger.info(f"Translated: {final[0]}")

            # Apply replacements
            for old, new in Utils.TAGS.items():
                final[0] = final[0].replace(old, new)

            return final[0]

        except Exception as e:
            logger.error(f"Translation failed: {e}", exc_info=True)
            return ""

    def _init_model(self):
        """Initialize model with optimized settings"""
        logger.info("Initializing model...")

        # Initialize processor and tags in the worker process
        self.processor = IndicProcessor(inference=True)
        self.tags = Utils.TAGS

        # Configure quantization
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
                bnb_8bit_quant_type="nf8",
            )
        else:
            qconfig = None

        # Load model with optimizations
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            quantization_config=qconfig,
            torch_dtype=torch.float16 if not qconfig else None,
            device_map="auto",
            attn_implementation="flash_attention_2" if torch.cuda.is_available() else None
        )

        # Fix weights warning
        self.model.tie_weights()

        if qconfig is None and torch.cuda.is_available():
            self.model = self.model.to("cuda")
            self.model.half()

        self.model = torch.compile(self.model.eval())
        logger.info("Model initialized successfully")

    def process_message(self,ch, method, properties, body):
        if self.producer is not None:
            try:
                body_str = gzip.decompress(body).decode("utf-8")  # RabbitMQ message body → str
                task_json = json.loads(body_str)

                # Dict → Result dataclass (with nested dataclasses)
                task = from_dict(Task, task_json)

                # task = pickle.loads(body)

                if not isinstance(task, Task):
                    raise TypeError(f"Expected Task, got {type(task)}")

                logger.info(f"Received task {task.id}, chunk {task.chunk_index + 1}/{task.total_chunks}")

                element = task.element

                if isinstance(element, Paragraph):
                    logger.info('Element instance of Paragraph')
                    self.translate_paragraph(element, task)
                elif isinstance(element, Table):
                    logger.info('Element instance of Table')
                    self.translate_table(element, task.language_config)

                result = Result(
                        id=task.id,
                        element=element,
                        language_config=task.language_config,
                        filename=task.filename,
                        chunk_index=task.chunk_index,
                        total_chunks = task.total_chunks,
                        meta_data= task.meta_data
                    )

                result_json = json.dumps(asdict(result)) # type: ignore[arg-type]

                logger.info(f'Publishing Result: {result}')
                compressed_result = gzip.compress(result_json.encode("utf-8"))
                self.producer.publish(compressed_result, persistent=False)

                # Acknowledge after processing
                ch.basic_ack(delivery_tag=method.delivery_tag)
                logger.info("Message acknowledged")
            except Exception as e:
                logger.error(f"Error: {e}", exc_info=True)
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

    def run(self):
        logger.info("[GPUWorker] Starting and loading model...")

        consumer = RabbitMQConsumer(host=Utils.KEY_RABBITMQ_HOST, queue=Utils.QUEUE_TASKS)
        consumer.connect()

        self.producer = RabbitMQProducer(host=Utils.KEY_RABBITMQ_HOST, queue=Utils.QUEUE_RESULTS)
        self.producer.connect()

        self._init_model()

        try:
            consumer.consume(callback=self.process_message)
        finally:
            consumer.close()
            self.producer.close()