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
        self._stop_event = False
        self.tokenizer = None
        self.model = None
        self.processor = IndicProcessor(inference=True)
        self.tags = Utils.TAGS

    def _init_model(self):
        """Initialize model with optimized settings"""
        logger.info("Initializing model...")

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

        if qconfig is None and torch.cuda.is_available():
            self.model = self.model.to("cuda")
            self.model.half()

        self.model = torch.compile(self.model.eval())
        logger.info("Model initialized successfully")

    def _process_batch(self, batch: List[Dict]) -> List[str]:
        """Improved batch processing using the better old approach"""
        texts = [item['text'] for item in batch]
        tgt_lang = batch[0]['lang']  # All items in batch share same language

        try:
            # 1. Preprocessing
            processed = self.processor.preprocess_batch(
                texts,
                src_lang='en_Latn',
                tgt_lang=tgt_lang
            )

            # 2. Tokenization
            inputs = self.tokenizer(
                processed,
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(self.model.device)

            # 3. Inference with GPU lock
            with torch.inference_mode():
                outputs = self.model.generate(
                        **inputs,
                        min_length=0,
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

            # 4. Decoding
            decoded = self.tokenizer.batch_decode(
                outputs.detach().cpu().tolist(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )

            # 5. Postprocessing
            final = self.processor.postprocess_batch(decoded, lang=tgt_lang)

            # 6. Apply any special replacements
            for i in range(len(final)):
                for old, new in self.tags.items():  # You'll need to define TAGS
                    final[i] = final[i].replace(old, new)

            return final

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            return [""] * len(batch)

    def run(self):
        """Main processing loop"""
        torch.set_num_threads(1)  # Critical for stability
        self._init_model()
        logger.info("GPU worker ready")

        while not self._stop_event:
            batch = []
            try:
                # Build batch
                while len(batch) < self.batch_size:
                    try:
                        task = self.task_queue.get(timeout=0.1)  # Non-blocking with timeout
                        batch.append(task)
                    except:
                        if batch:  # Process partial batch if items exist
                            break
                        time.sleep(0.01)  # Prevent busy waiting
                        continue

                if batch:
                    # Process and return results
                    translations = self._process_batch(batch)
                    for task, translation in zip(batch, translations):
                        task['result_queue'].put(translation)

            except Exception as e:
                logger.error(f"GPU worker error: {e}")

    def stop(self):
        """Graceful shutdown"""
        self._stop_event = True
        if self.model:
            del self.model
            torch.cuda.empty_cache()