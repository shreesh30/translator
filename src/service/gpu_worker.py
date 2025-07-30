import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig
from multiprocessing import Process, Queue
import logging
import queue

logger = logging.getLogger(__name__)


class GPUWorker(Process):
    def __init__(self, model_name: str, task_queue: Queue, result_queue: Queue, quantization: str = None):
        super().__init__(daemon=True)
        self.model_name = model_name
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.quantization = quantization
        self._stop_flag = False
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name,trust_remote_code=True)
        self.model = self.load_model()

    def load_model(self):
        # Your exact quantization logic
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

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            quantization_config=qconfig,
            torch_dtype=torch.float16 if not qconfig else None
        )

        if qconfig is None:
            self.model = self.model.to("cuda")
            self.model.half()  # Only needed if not quantized

        # Add optimizations (optional)
        model = torch.compile(self.model.eval())

        return model

    def run(self):
        logger.info("[GPUWorker] Started and ready to receive tasks")

        while not self._stop_flag:
            try:
                tasks = []

                # Collect up to 10 tasks or break early if queue is temporarily empty
                for _ in range(10):
                    try:
                        task = self.task_queue.get(timeout=0.2)  # slight wait to prevent busy looping
                        tasks.append(task)
                    except queue.Empty:
                        break

                if not tasks:
                    continue

                # Validate language consistency
                target_langs = {t['lang'] for t in tasks}
                if len(target_langs) > 1:
                    logger.warning(f"[GPUWorker] Mixed languages in batch: {target_langs}")
                    continue  # Skip batch or handle separately if needed

                texts = [t['text'] for t in tasks]
                lang = tasks[0]['lang']

                # Tokenize and move to GPU
                inputs = self.tokenizer(
                    texts,
                    padding="longest",
                    truncation=True,
                    return_tensors="pt"
                ).to("cuda")

                with torch.inference_mode():
                    outputs = self.model.generate(
                        **inputs,
                        min_length=0,
                        max_length=1024,
                        num_beams=6,
                        length_penalty=1.0,
                        early_stopping=True,
                        repetition_penalty=1.1,
                        do_sample=True,
                        temperature=0.7,
                        no_repeat_ngram_size=3,
                        use_cache=False
                    )

                results = self.tokenizer.batch_decode(
                    outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )

                for task, result in zip(tasks, results):
                    try:
                        task['result_queue'].put(result)
                    except Exception as e:
                        logger.error(f"[GPUWorker] Failed to deliver result: {e}")

            except Exception as e:
                logger.exception(f"[GPUWorker] Unexpected error: {e}")

    def stop(self):
        self._stop_flag = True