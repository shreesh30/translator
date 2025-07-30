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


def setup_console_logging(log_file_path='logs/output.log'):
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file_path)
        ]
    )


set_start_method('spawn', force=True)  # Critical for CUDA compatibility


def run_pipeline(lang_configs: List[LanguageConfig], input_path: str, output_path: str):
    """Optimal parallel processing with 1 GPU worker + multi-threaded CPU workers"""
    # 1. Create communication queues in main process
    gpu_task_queue = Queue(maxsize=100)
    gpu_result_queue = Queue()

    # 2. Start single GPU worker process
    gpu_worker = Process(
        target=run_gpu_worker,
        args=(gpu_task_queue, gpu_result_queue),
        daemon=True
    )
    gpu_worker.start()

    try:
        # 3. Process languages in parallel threads (not processes)
        with ThreadPoolExecutor(max_workers=len(lang_configs)) as executor:
            futures = []
            for config in lang_configs:
                futures.append(executor.submit(
                    process_language,
                    config,
                    input_path,
                    output_path,
                    gpu_task_queue,
                    gpu_result_queue
                ))

            for future in futures:
                future.result()  # Wait for completion
    finally:
        gpu_worker.terminate()

def run_gpu_worker(task_queue, result_queue):
    """Dedicated GPU process"""
    worker = GPUWorker(
        model_name="ai4bharat/indictrans2-en-indic-1B",
        task_queue=task_queue,
        result_queue=result_queue,
        quantization="8-bit"
    )
    worker.run()

def process_language(config, input_path, output_path, task_queue, result_queue):
    """Thread-safe language processor"""
    processor = PDFProcessor(
        lang_config=config,
        gpu_task_queue=task_queue,
        gpu_result_queue=result_queue,
        input_path=input_path,
        output_path=output_path
    )
    processor.run()


if __name__ == "__main__":
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