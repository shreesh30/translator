from src.translate_pdf import process_pdf

# from src.translate_pdf_test import process_pdf

if __name__ == "__main__":
    # input_pdf_path = "/Users/shreesharya/Documents/Development/Translator/resource/input/2-pages.pdf"               # Replace with your file
    # input_pdf_path = "/Users/shreesharya/Documents/Development/Translator/resource/input/table-single-page.pdf"               # Replace with your file
    input_pdf_path = "/Users/shreesharya/Documents/Development/Translator/resource/input/original-doc-without-intro.pdf"               # Replace with your file
    # input_pdf_path = "/Users/shreesharya/Documents/Development/Translator/resource/input/para-with-dots-2.pdf"               # Replace with your file
    output_pdf_path = "/Users/shreesharya/Documents/Development/Translator/resource/table_translated.pdf"  # Output filename
    src_lang = "eng_Latn"
    # tgt_lang = "ben_Beng"
    tgt_lang = "hin_Deva"

    # main_execution(input_pdf_path, src_lang, tgt_lang)
    # process_pdf(input_pdf_path,output_pdf_path, src_lang, tgt_lang)
    process_pdf(input_pdf_path,output_pdf_path, src_lang, tgt_lang)
