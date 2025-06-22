from src.model.language_config import LanguageConfig
from src.translate_pdf import PDFTranslator


if __name__ == "__main__":
    input_pdf_path = "resource/input/cad_09-12-1946_pages_16_to_23-1.pdf"               # Replace with your file
    output_pdf_path = "/Users/shreesharya/Documents/Development/Translator/resource/output/translated.pdf"  # Output filename

    input_file = "src/IndicTrans/train_data/multi_lang_tagged_train.tsv"
    output_file = "src/IndicTrans/train_data/multi_lang_tagged_train_translated.tsv"

    language_configs = [
        # LanguageConfig("Odia", "ory_Orya", "resource/fonts/SakalBharati_N_Ship.ttf", 1.2),
        LanguageConfig("Bengali","ben_Beng", "resource/fonts/bnotdurga_n_ship.ttf", 1.1, 1.3)
    ]

    pdf_translator=None
    for lang_config in language_configs:
        pdf_translator=PDFTranslator(lang_config)
        pdf_translator.process_pdf(input_folder_path=input_pdf_path,output_folder_path='resource/tmp/docx')
