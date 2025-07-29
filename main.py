import logging
import os

from src.model.language_config import LanguageConfig
from src.service.pdf_translator import PDFTranslator
from multiprocessing import Process, Lock
import multiprocessing as mp

# def setup_console_logging():
#     """Configure logging to print only to console."""
#     logging.basicConfig(
#         level=logging.INFO,  # Set default level (INFO or DEBUG)
#         format='%(name)s - %(levelname)s - %(message)s',
#     )

def setup_console_logging(log_file_path='logs/output.log'):
    """Configure logging to print to console and save to a file."""
    # Ensure the log directory exists
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    # Clear existing handlers if re-running the setup
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Console output
            logging.FileHandler(log_file_path)  # File output
        ]
    )

def translate_for_language(lang_config: LanguageConfig, input_path: str, output_path: str, gpu_lock: Lock):
    setup_console_logging()
    try:
        logging.info(f"Starting translation for: {lang_config.get_target_language()}")
        translator = PDFTranslator(lang_config, quantization="8-bit")
        # Inject the shared lock into the class (if needed)
        PDFTranslator.GPU_LOCK = gpu_lock
        translator.process_pdf(input_folder_path=input_path, output_folder_path=output_path)
        logging.info(f"Completed translation for: {lang_config.get_target_language()}")
    except Exception as e:
        logging.error(f"Failed translation for {lang_config.get_target_language()}: {e}")

def run_parallel_translation(lang_configs, input_path: str, output_path: str):
    gpu_lock = Lock()  # Shared GPU lock
    processes = []

    for config in lang_configs:
        process = Process(
            target=translate_for_language,
            args=(config, input_path, output_path, gpu_lock),
            name=f"Translator-{config.get_target_language()}"
        )
        processes.append(process)
        process.start()

    for p in processes:
        p.join()


if __name__ == "__main__":
    setup_console_logging()
    # input_pdf_path = "resource/input/cad_21-07-1947.pdf.DEBATE27-4-12-1-2.pdf"  # Replace with your file
    # input_pdf_path = "resource/input/cad_21-07-1947.pdf.DEBATE27-4-12-1-3.pdf"  # Replace with your file
    # input_pdf_path = "resource/input/cad_21-07-1947.pdf.DEBATE27-4-12.pdf"               # Replace with your file
    # input_pdf_path = "resource/input/cad_09-12-1946_pages_16_to_23.pdf"               # Replace with your file
    # input_pdf_path = "resource/input/original-doc-16-17.pdf"               # Replace with your file
    # input_pdf_path = "resource/input/table-1.pdf"               # Replace with your file
    input_pdf_path = "resource/input/pdf"               # Replace with your file
    # input_pdf_path = "resource/input/table-2-page.pdf"               # Replace with your file
    # input_pdf_path = "resource/input/cad_04-11-1948.pdf.DEBATE 48.pdf"               # Replace with your file
    # input_pdf_path = "resource/input/preface-og.pdf"               # Replace with your file
    # input_pdf_path = "resource/input/original-doc.pdf"               # Replace with your file
    # input_pdf_path = "resource/input/original-doc-4.pdf"               # Replace with your file
    # input_pdf_path = "resource/input/cad_09-12-1946_pages_16_to_23-1.pdf"               # Replace with your file
    output_pdf_path = "resource/output"  # Output filename

    input_file = "src/IndicTrans/train_data/multi_lang_tagged_train.tsv"
    output_file = "src/IndicTrans/train_data/multi_lang_tagged_train_translated.tsv"

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
    mp.set_start_method("spawn", force=True)
    run_parallel_translation(language_configs, input_pdf_path, output_pdf_path)
