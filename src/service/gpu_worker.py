from threading import Event

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig
from multiprocessing import Process, Queue
from IndicTransToolkit.processor import IndicProcessor
import logging
import queue
from typing import List, Dict
import time

from src.utils.utils import Utils

logger = logging.getLogger(__name__)


class GPUWorker(Process):
    def __init__(self,model_name: str,task_queue: Queue,result_queue: Queue,quantization: str = "8-bit",batch_size: int = 10,max_seq_length: int = 1024):
        super().__init__(daemon=True)
        self.model_name = model_name
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.quantization = quantization
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self._stop_event = Event()
        self.tokenizer = None
        self.model = None
        self.processor = None
        self.tags = None

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

    def run(self):
        """Main processing loop"""
        torch.set_num_threads(1)
        self._init_model()
        logger.info("GPU worker ready")

        while not self._stop_event.is_set():
            try:
                task = self.task_queue.get(timeout=0.1)

                # Process the text
                translation = self._process_item(task)

                # Send result back through the provided queue
                task['result_queue'].put(translation)

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"GPU worker error: {str(e)}", exc_info=True)
                time.sleep(0.1)

    def _process_item(self, task: Dict) -> str:
        """Process single translation item"""
        try:
            # Preprocessing
            processed = self.processor.preprocess_batch(
                [task['text']],
                src_lang='eng_Latn',
                tgt_lang=task['lang']
            )[0]

            # Tokenization
            inputs = self.tokenizer(
                processed,
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(self.model.device)

            # Inference
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.max_seq_length,
                    num_beams=6,
                    length_penalty=1.0,
                    early_stopping=True,
                    repetition_penalty=1.1,
                    do_sample=True,
                    temperature=0.7,
                    no_repeat_ngram_size=3,
                    use_cache=False
                )

            # Decoding
            decoded = self.tokenizer.batch_decode(
                outputs,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

            # Postprocessing
            final = self.processor.postprocess_batch([decoded], lang=task['lang'])[0]

            # Apply replacements
            for old, new in self.tags.items():
                final = final.replace(old, new)

            return final

        except Exception as e:
            logger.error(f"Translation failed: {str(e)}")
            return ""

    def stop(self):
        """Graceful shutdown"""
        self._stop_event = True
        if self.model:
            del self.model
            torch.cuda.empty_cache()