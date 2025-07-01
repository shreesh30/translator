from src.model.language_config import LanguageConfig
from src.translate_pdf import PDFTranslator

if __name__ == "__main__":
    # input_pdf_path = "resource/input/cad_21-07-1947.pdf.DEBATE27-4-12-1-2.pdf"  # Replace with your file
    # input_pdf_path = "resource/input/cad_21-07-1947.pdf.DEBATE27-4-12.pdf"               # Replace with your file
    # input_pdf_path = "resource/input/cad_09-12-1946_pages_16_to_23.pdf"               # Replace with your file
    input_pdf_path = "resource/input/original-doc-16-17.pdf"               # Replace with your file
    output_pdf_path = "/Users/shreesharya/Documents/Development/Translator/resource/output/translated.pdf"  # Output filename

    input_file = "src/IndicTrans/train_data/multi_lang_tagged_train.tsv"
    output_file = "src/IndicTrans/train_data/multi_lang_tagged_train_translated.tsv"

    language_configs = [
        # LanguageConfig(target_language="Odia", target_language_key="ory_Orya",
        #                target_font_path="resource/fonts/SakalBharati_N_Ship.ttf", font_multiplier=1.1,
        #                line_spacing_multiplier=1.25),
        # LanguageConfig(target_language="Bengali", target_language_key="ben_Beng",
        #                target_font_path="resource/fonts/bnotdurga_n_ship.ttf", font_multiplier=1.1,
        #                line_spacing_multiplier=1.25),
        LanguageConfig(target_language="Hindi", target_language_key="hin_Deva",
                       target_font_path="resource/fonts/DVOTSurekh_N_Ship.ttf", font_multiplier=1.1,
                       line_spacing_multiplier=1.25),
        # LanguageConfig(target_language="Dogri", target_language_key="doi_Deva",
        #                target_font_path="resource/fonts/SakalBharati_N_Ship.ttf", font_multiplier=1.1,
        #                line_spacing_multiplier=1.25),
        # LanguageConfig(target_language="Gujarati", target_language_key="guj_Gujr",
        #                target_font_path="resource/fonts/GJOTAvantika_N_Ship.ttf", font_multiplier=1.1,
        #                line_spacing_multiplier=1.25),
        # LanguageConfig(target_language="Malayalam", target_language_key="mal_Mlym",
        #                target_font_path="resource/fonts/MLOT-Karthika_N_Ship.ttf", font_multiplier=1.1,
        #                line_spacing_multiplier=1.25),
        # LanguageConfig(target_language="Marathi", target_language_key="mar_Deva",
        #                target_font_path="resource/fonts/Sakal Marathi Normal_Ship.ttf", font_multiplier=1.1,
        #                line_spacing_multiplier=1.25),
        # LanguageConfig(target_language="Punjabi", target_language_key="pan_Guru",
        #                target_font_path="resource/fonts/PNOTAmar_N_Ship.ttf", font_multiplier=1.1,
        #                line_spacing_multiplier=1.25),
        # LanguageConfig(target_language="Kannada", target_language_key="kan_Knda",
        #                target_font_path="resource/fonts/SakalBharati_N_Ship.ttf", font_multiplier=1.1,
        #                line_spacing_multiplier=1.25),
        # LanguageConfig(target_language="Assamese", target_language_key="asm_Beng",
        #                target_font_path="resource/fonts/SakalBharati_N_Ship.ttf", font_multiplier=1.1,
        #                line_spacing_multiplier=1.25),
        # LanguageConfig(target_language="Bodo", target_language_key="brx_Deva",
        #                target_font_path="resource/fonts/SakalBharati_N_Ship.ttf", font_multiplier=1.1,
        #                line_spacing_multiplier=1.25),
        # LanguageConfig(target_language="Konkani", target_language_key="gom_Deva",
        #                target_font_path="resource/fonts/SakalBharati_N_Ship.ttf", font_multiplier=1.1,
        #                line_spacing_multiplier=1.25),
        # LanguageConfig(target_language="Maithili", target_language_key="mai_Deva",
        #                target_font_path="resource/fonts/SakalBharati_N_Ship.ttf", font_multiplier=1.1,
        #                line_spacing_multiplier=1.25),
        # LanguageConfig(target_language="Nepali", target_language_key="npi_Deva",
        #                target_font_path="resource/fonts/SakalBharati_N_Ship.ttf", font_multiplier=1.1,
        #                line_spacing_multiplier=1.25),
        # LanguageConfig(target_language="Sanskrit", target_language_key="san_Deva",
        #                target_font_path="resource/fonts/SakalBharati_N_Ship.ttf", font_multiplier=1.1,
        #                line_spacing_multiplier=1.25),
        # LanguageConfig(target_language="Santhali", target_language_key="sat_Olck",
        #                target_font_path="resource/fonts/SakalBharati_N_Ship.ttf", font_multiplier=1.1,
        #                line_spacing_multiplier=1.25),
        # LanguageConfig(target_language="Tamil", target_language_key="tam_Taml",
        #                target_font_path="resource/fonts/SakalBharati_N_Ship.ttf", font_multiplier=1.1,
        #                line_spacing_multiplier=1.25),
        # LanguageConfig(target_language="Telugu", target_language_key="tel_Telu",
        #                target_font_path="resource/fonts/SakalBharati_N_Ship.ttf", font_multiplier=1.1,
        #                line_spacing_multiplier=1.25),
        # LanguageConfig(target_language="Urdu", target_language_key="urd_Arab",
        #                target_font_path="resource/fonts/UROT-Ghalib_N_Ship.ttf", font_multiplier=1.1,
        #                line_spacing_multiplier=1.25, right_to_left=True),
        # LanguageConfig(target_language="Kashmiri", target_language_key="kas_Arab",
        #                target_font_path="resource/fonts/UROT-Ghalib_N_Ship.ttf", font_multiplier=1.1,
        #                line_spacing_multiplier=1.25, right_to_left=True)

    ]

    pdf_translator = None
    for lang_config in language_configs:
        pdf_translator = PDFTranslator(lang_config)
        pdf_translator.process_pdf(input_folder_path=input_pdf_path, output_folder_path='resource/tmp/docx')
