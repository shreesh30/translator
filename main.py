import logging
import os
import platform
import subprocess
from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures.thread import ThreadPoolExecutor
from string import Template

from src.model.language_config import LanguageConfig
from src.service.gpu_worker import GPUWorker
from src.service.pdf_processor import PDFProcessor
from src.service.pdf_translator import PDFTranslator
from multiprocessing import Process, Lock, Queue,set_start_method
from typing import List
import multiprocessing as mp

from src.service.result_handler import ResultHandler


'''def setup_logging(log_file_name: str):
    """Configure logging to a file for the current process."""
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        filename=os.path.join("logs", log_file_name),
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(processName)s - %(message)s",
        force=True  # override inherited loggers
    )

def run_pipeline(lang_configs, input_path, output_path):
    logger = logging.getLogger("main")
    logger.info("Initializing queues and starting processes...")

    # Create queues
    gpu_task_queue = Queue(maxsize=100)
    gpu_result_queue = Queue()

    # Start GPU Worker process
    gpu_worker = Process(
        target=run_gpu_worker,
        args=(gpu_task_queue, gpu_result_queue)
    )
    gpu_worker.start()

    # Start Result Handler process
    result_handler = Process(
        target=run_result_handler,
        args=(gpu_result_queue, output_path)
    )
    result_handler.start()

    # Start processing PDFs
    logger.info("Starting PDF processing with input path: %s", input_path)
    processor = PDFProcessor(
        lang_configs=lang_configs,
        input_path=input_path,
        gpu_task_queue=gpu_task_queue
    )
    processor.process_all_pdfs()

    # Wait for child processes (optional, or use daemon=True for fire-and-forget)
    gpu_worker.join()
    result_handler.join()

def run_gpu_worker(task_queue: Queue, result_queue: Queue):
    setup_logging("gpu_worker.log")
    logger = logging.getLogger("gpu_worker")
    logger.info("GPU Worker process started.")

    worker = GPUWorker(
        model_name="ai4bharat/indictrans2-en-indic-1B",
        input_queue=task_queue,
        output_queue=result_queue,
        quantization="8-bit"
    )
    worker.run()

def run_result_handler(result_queue: Queue, output_path):
    setup_logging("result_handler.log")
    logger = logging.getLogger("result_handler")
    logger.info("Result Handler process started.")

    handler = ResultHandler(result_queue, output_path)
    handler.run()'''

# Path to the service template
TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "templates", "service_template.service")
GENERATED_DIR = os.path.join(os.path.dirname(__file__), "generated_services")

def generate_service_file(service_name, description, user, working_directory, exec_start):
    # Load the template file
    with open(TEMPLATE_PATH, "r") as f:
        template_content = Template(f.read())

    # Replace placeholders with actual values
    service_content = template_content.substitute(
        description=description,
        user=user,
        working_directory=working_directory,
        exec_start=exec_start
    )

    # Choose where to place the file based on OS
    if platform.system() == "Linux":
        output_path = f"/etc/systemd/system/{service_name}.service"
    else:
        os.makedirs(GENERATED_DIR, exist_ok=True)
        output_path = os.path.join(GENERATED_DIR, f"{service_name}.service")

    # Write the final service file
    with open(output_path, "w") as f:
        f.write(service_content)

    print(f"Service file generated at: {output_path}")
    return output_path

def install_service(service_name):
    if platform.system() == "Linux":
        subprocess.run(["systemctl", "daemon-reload"], check=True)
        subprocess.run(["systemctl", "enable", service_name], check=True)
        subprocess.run(["systemctl", "start", service_name], check=True)
        print(f"Service '{service_name}' installed and started.")
    else:
        print(f"On macOS: Service file saved to generated_services/, not installed.")


if __name__ == "__main__":
    # try:
    #     set_start_method("spawn", force=True)
    # except RuntimeError:
    #     pass
    #
    # setup_logging("output.log")
    # input_pdf_path = "resource/input/pdf-complete"               # Replace with your file
    # # input_pdf_path = "resource/tmp"               # Replace with your file
    # output_pdf_path = "resource/output"  # Output filename
    #
    # language_configs = [
        # LanguageConfig(target_language="Odia", target_language_key="ory_Orya",
        #                target_font_path="resource/fonts/SakalBharati_N_Ship.ttf", font_size_multiplier=1.1,
        #                line_spacing_multiplier=1.25),
        # LanguageConfig(target_language="Bengali", target_language_key="ben_Beng",
        #                target_font_path="resource/fonts/bnotdurga_n_ship.ttf", font_size_multiplier=1.1,
        #                line_spacing_multiplier=1.25),
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

    # ]
    # run_pipeline(language_configs, input_pdf_path, output_pdf_path)

    services_to_create = [
        {
            "name": "ingestion_service",
            "description": "Ingestion Service",
            "script": "ingestion_service/main.py"
        },
        {
            "name": "orchestration_service",
            "description": "Orchestration Service",
            "script": "orchestration_service/main.py"
        }
    ]

    for svc in services_to_create:
        working_dir = os.path.join(os.path.dirname(__file__), "src")
        exec_command = f"python3 {os.path.join(working_dir, svc['script'])}"

        path = generate_service_file(
            service_name=svc["name"],
            description=svc["description"],
            user=os.getenv("USER"),
            working_directory=working_dir,
            exec_start=exec_command
        )

        install_service(svc["name"])