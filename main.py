import logging
import os
from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures.thread import ThreadPoolExecutor

from src.model.language_config import LanguageConfig
from src.service.gpu_worker import GPUWorker
from src.service.pdf_processor import PDFProcessor
from src.service.pdf_translator import PDFTranslator
from multiprocessing import Process, Lock, Queue,set_start_method
from typing import List
import multiprocessing as mp

from src.service.result_handler import ResultHandler


def setup_console_logging(log_file_path='logs/output.log'):
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file_path, mode='w')  # Overwrite on each main run
        ]
    )

def run_pipeline(lang_configs: List[LanguageConfig], input_path: str, output_path: str):
    # Create queues
    gpu_task_queue = Queue(maxsize=100)
    gpu_result_queue = Queue()

    # Start GPU worker process
    gpu_worker = Process(
        target=run_gpu_worker,
        args=(gpu_task_queue, gpu_result_queue),
    )
    gpu_worker.start()

    # Start result handler thread
    result_handler = ResultHandler(
        output_queue=gpu_result_queue,
        output_path=output_path,
    )
    result_handler.start()

    try:
        # Start single PDF processor to enqueue tasks
        processor = PDFProcessor(
            lang_configs=lang_configs,
            gpu_task_queue=gpu_task_queue,
            input_path=input_path
        )
        processor.process_all_pdfs()

    finally:
        gpu_worker.terminate()
        result_handler.stop()

def run_gpu_worker(task_queue, result_queue):
    """Dedicated GPU process"""
    import logging


    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "gpu_worker.log")

    logger = logging.getLogger("gpu_worker")
    logger.setLevel(logging.INFO)

    # Clear old handlers if any
    if logger.hasHandlers():
        logger.handlers.clear()

    fh = logging.FileHandler(log_file, mode="a")
    fh.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)

    logger.addHandler(fh)

    logger = logging.getLogger(__name__)
    logger.info("GPU worker started")
    try:
        worker = GPUWorker(
            model_name="ai4bharat/indictrans2-en-indic-1B",
            input_queue=task_queue,
            output_queue=result_queue,
            quantization="8-bit"
        )
        worker.run()
    except Exception as e:
        logger.exception(f"GPU Worker crashed: {e}")
#
# def process_language(config, input_path, output_path, task_queue, result_queue):
#     """Thread-safe language processor"""
#     processor = PDFProcessor(
#         lang_config=config,
#         gpu_task_queue=task_queue,
#         gpu_result_queue=result_queue,
#         input_path=input_path,
#         output_path=output_path
#     )
#     processor.run()

if __name__ == "__main__":
    set_start_method('spawn', force=True)  # Critical for CUDA compatibility

    setup_console_logging()
    input_pdf_path = "resource/input/pdf"               # Replace with your file
    # input_pdf_path = "resource/tmp"               # Replace with your file
    output_pdf_path = "resource/output"  # Output filename

    language_configs = [
        # LanguageConfig(target_language="Odia", target_language_key="ory_Orya",
        #                target_font_path="resource/fonts/SakalBharati_N_Ship.ttf", font_size_multiplier=1.1,
        #                line_spacing_multiplier=1.25),
        LanguageConfig(target_language="Bengali", target_language_key="ben_Beng",
                       target_font_path="resource/fonts/bnotdurga_n_ship.ttf", font_size_multiplier=1.1,
                       line_spacing_multiplier=1.25),
        # LanguageConfig(target_language="Hindi", target_language_key="hin_Deva",
        #                target_font_path="resource/fonts/DVOTSurekh_N_Ship.ttf", font_size_multiplier=1.1,
        #                line_spacing_multiplier=1.25),
        # LanguageConfig(target_language="Dogri", target_language_key="doi_Deva",
        #                target_font_path="resource/fonts/SakalBharati_N_Ship.ttf", font_size_multiplier=1.1,
        #                line_spacing_multiplier=1.25),
        # LanguageConfig(target_language="Gujarati", target_language_key="guj_Gujr",
        #                target_font_path="resource/fonts/GJOTAvantika_N_Ship.ttf", font_size_multiplier=1.1,
        #                line_spacing_multiplier=1.25),
        # LanguageConfig(target_language="Malayalam", target_language_key="mal_Mlym",
        #                target_font_path="resource/fonts/MLOT-Karthika_N_Ship.ttf", font_size_multiplier=1.1,
        #                line_spacing_multiplier=1.25),
        # LanguageConfig(target_language="Marathi", target_language_key="mar_Deva",
        #                target_font_path="resource/fonts/Sakal Marathi Normal_Ship.ttf", font_size_multiplier=1.1,
        #                line_spacing_multiplier=1.25),
        # LanguageConfig(target_language="Punjabi", target_language_key="pan_Guru",
        #                target_font_path="resource/fonts/PNOTAmar_N_Ship.ttf", font_size_multiplier=1.1,
        #                line_spacing_multiplier=1.25),
        # LanguageConfig(target_language="Kannada", target_language_key="kan_Knda",
        #                target_font_path="resource/fonts/SakalBharati_N_Ship.ttf", font_size_multiplier=1.1,
        #                line_spacing_multiplier=1.25),
        # LanguageConfig(target_language="Assamese", target_language_key="asm_Beng",
        #                target_font_path="resource/fonts/SakalBharati_N_Ship.ttf", font_size_multiplier=1.1,
        #                line_spacing_multiplier=1.25),
        # LanguageConfig(target_language="Bodo", target_language_key="brx_Deva",
        #                target_font_path="resource/fonts/SakalBharati_N_Ship.ttf", font_size_multiplier=1.1,
        #                line_spacing_multiplier=1.25),
        # LanguageConfig(target_language="Konkani", target_language_key="gom_Deva",
        #                target_font_path="resource/fonts/SakalBharati_N_Ship.ttf", font_size_multiplier=1.1,
        #                line_spacing_multiplier=1.25),
        # LanguageConfig(target_language="Maithili", target_language_key="mai_Deva",
        #                target_font_path="resource/fonts/SakalBharati_N_Ship.ttf", font_size_multiplier=1.1,
        #                line_spacing_multiplier=1.25),
        # LanguageConfig(target_language="Nepali", target_language_key="npi_Deva",
        #                target_font_path="resource/fonts/SakalBharati_N_Ship.ttf", font_size_multiplier=1.1,
        #                line_spacing_multiplier=1.25),
        # LanguageConfig(target_language="Sanskrit", target_language_key="san_Deva",
        #                target_font_path="resource/fonts/SakalBharati_N_Ship.ttf", font_size_multiplier=1.1,
        #                line_spacing_multiplier=1.25),
        # LanguageConfig(target_language="Santhali", target_language_key="sat_Olck",
        #                target_font_path="resource/fonts/SakalBharati_N_Ship.ttf", font_size_multiplier=1.1,
        #                line_spacing_multiplier=1.25),
        # LanguageConfig(target_language="Tamil", target_language_key="tam_Taml",
        #                target_font_path="resource/fonts/SakalBharati_N_Ship.ttf", font_size_multiplier=1.1,
        #                line_spacing_multiplier=1.25),
        # LanguageConfig(target_language="Telugu", target_language_key="tel_Telu",
        #                target_font_path="resource/fonts/SakalBharati_N_Ship.ttf", font_size_multiplier=1.1,
        #                line_spacing_multiplier=1.25),
        # LanguageConfig(target_language="Urdu", target_language_key="urd_Arab",
        #                target_font_path="resource/fonts/UROT-Ghalib_N_Ship.ttf", font_size_multiplier=1.1,
        #                line_spacing_multiplier=1.25, right_to_left=True),
        # LanguageConfig(target_language="Kashmiri", target_language_key="kas_Arab",
        #                target_font_path="resource/fonts/UROT-Ghalib_N_Ship.ttf", font_size_multiplier=1.1,
        #                line_spacing_multiplier=1.25, right_to_left=True)

    ]

    # pdf_translator = None
    # for lang_config in language_configs:
    #     pdf_translator = PDFTranslator(lang_config)
    #     pdf_translator.process_pdf(input_folder_path=input_pdf_path, output_folder_path='resource/output')
    run_pipeline(language_configs, input_pdf_path, output_pdf_path)