import re
import sys

import torch
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches
from fontTools.ttLib import TTFont as FontToolsTTFont  # For font metadata
from pdf2docx import Converter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont as ReportlabTTFont  # For PDF registration
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig

from IndicTransToolkit.IndicTransToolkit import IndicProcessor

# ─── Configuration ─────────────────────────────────────────────────────────────
BATCH_SIZE = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
quantization = None

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

FONT_PATH = "/Users/shreesharya/Documents/Development/Translator/resource/fonts/DVOTSurekh_N_Ship.ttf"
FONT_NAME = None

# Get actual font name using fontTools
try:
    tt = FontToolsTTFont(FONT_PATH)
    actual_font_name = tt['name'].getDebugName(1)  # Get font's internal name
    FONT_NAME = actual_font_name.split('-')[0]  # Usually "NotoSansDevanagari"
    tt.close()
    print(f"Detected font name: {FONT_NAME}")
except Exception as e:
    print(f"Font metadata read failed: {str(e)}")
    raise

# Register font with ReportLab
try:
    pdfmetrics.registerFont(ReportlabTTFont(FONT_NAME, FONT_PATH))
    print(f"Successfully registered font: {FONT_NAME}")
except Exception as e:
    print(f"Font registration failed: {str(e)}")
    raise

# try:
#     nltk.data.find('tokenizers/punkt')
# except LookupError:
#     nltk.download('punkt')
#
# punkt_param = PunktParameters()
# punkt_tokenizer = PunktSentenceTokenizer(punkt_param)

def batch_translate(elements, src_lang, tgt_lang, model, tokenizer, ip):
    """Translates all content while preserving structure"""
    # Extract ALL content for translation
    to_translate = [elem['content'] for elem in elements]
    total = len(to_translate)
    print(f"[INFO] Translating {total} elements...")

    translations = []
    for i in range(0, total, BATCH_SIZE):
        batch = to_translate[i:i + BATCH_SIZE]
        print(f"\n[INFO] Processing batch {i // BATCH_SIZE + 1} ({i} to {min(i + BATCH_SIZE, total)})")

        try:
            # Add language tags and preprocess
            # tagged_batch = [f"<2{tgt_lang}> {text}" for text in batch]
            processed_batch = ip.preprocess_batch(batch, src_lang=src_lang, tgt_lang=tgt_lang)

            # Tokenization and generation
            try:
                inputs = tokenizer(
                    processed_batch,
                    truncation=True,
                    padding="longest",
                    return_tensors="pt",
                ).to(DEVICE)
                print(f"[DEBUG] Tokenized Inputs: {inputs}")
            except Exception as e:
                print(f"[ERROR] Tokenization failed: {e}")
                continue

            try:
                with torch.no_grad():
                    generated_tokens = model.generate(
                        **inputs,
                        min_length=0,
                        max_length=512,
                        num_beams=4,
                        early_stopping=True
                    )
                print(f"[DEBUG] Generated Token IDs: {generated_tokens}")
            except Exception as e:
                print(f"[ERROR] Generation failed: {e}")
                continue

            # Decoding and postprocessing
            try:
                with tokenizer.as_target_tokenizer():
                    decoded = tokenizer.batch_decode(
                        generated_tokens.detach().cpu().tolist(),
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                print(f"[DEBUG] Decoded Text: {decoded}")
            except Exception as e:
                print(f"[ERROR] Decoding failed: {e}")
                continue

            try:
                final_translations = ip.postprocess_batch(decoded, lang=tgt_lang)
                print(f"[DEBUG] Postprocessed Translations: {final_translations}")
                translations += final_translations
            except Exception as e:
                print(f"[ERROR] Postprocessing failed: {e}")
                continue

        except Exception as e:
            print(f"Error in batch {i // BATCH_SIZE + 1}: {str(e)}")
            translations.extend([""] * len(batch))  # Maintain structure
            continue

        del inputs
        torch.cuda.empty_cache()
    # Update ALL elements with translations
    for elem, translated_text in zip(elements, translations):
        elem['content'] = translated_text

    print(f"====================================elements: {elements}")

    return translations[0]

def initialize_model_and_tokenizer(ckpt_dir, quantization):
    if quantization == "4-bit":
        qconfig = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif quantization == "8-bit":
        qconfig = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
        )
    else:
        qconfig = None

    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        ckpt_dir,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        quantization_config=qconfig,
    )

    if qconfig == None:
        model = model.to(DEVICE)
        if DEVICE == "cuda":
            model.half()

    model.eval()

    return tokenizer, model

def resize_embeddings(embed_layer, new_size):
    old_weights = embed_layer.weight.data
    # Initialize new embeddings with random values
    new_weights = torch.randn(new_size, old_weights.shape[1])
    # Copy old embeddings into the new tensor
    new_weights[:old_weights.shape[0]] = old_weights
    # Update the embedding layer
    embed_layer.weight = torch.nn.Parameter(new_weights)
    return embed_layer


en_indic_ckpt_dir = "ai4bharat/indictrans2-en-indic-1B"
en_indic_tokenizer, en_indic_model = initialize_model_and_tokenizer(en_indic_ckpt_dir, quantization)
ip = IndicProcessor(inference=True)

def process_pdf(input_path, output_path, src_lang, tgt_lang):
    tmp_docx = "/Users/shreesharya/Documents/Development/Translator/resource/tmp/docx/temp" + ".intermediate.docx"
    cv = Converter(input_path)
    cv.convert(tmp_docx, start=0, end=None)
    cv.close()

    doc = Document(tmp_docx)

    for section in doc.sections:
        section.top_margin = Inches(0)
        section.bottom_margin = Inches(0)

    def restore_dots(translated_text, dot_counts):
        """Restores original dot sequences from [000] placeholders"""
        parts = translated_text.split("[000]")
        restored_text = []

        for i, part in enumerate(parts):
            restored_text.append(part)
            if i < len(dot_counts):
                restored_text.append("." * dot_counts[i])

        return "".join(restored_text)

    def clean_text(text):
        """Replace only sequences with ≥2 dots with [000]"""
        # Preserve numeric patterns
        text = re.sub(r'(\d+)([a-z]+)', r'\1 \2', text)  # Separate numbers and text
        text = re.sub(r'(\d+)(st|nd|rd|th)', r'\1\2', text)  # Keep ordinals intact

        cleaned_parts = []
        dot_counts = []

        # Split on sequences of 2+ dots
        for segment in re.split(r'(\.{2,})', text):
            if segment.startswith('.') and all(c == '.' for c in segment):
                # Only replace if entire segment is dots (2+)
                dot_counts.append(len(segment))
                cleaned_parts.append("[000]")
            else:
                cleaned_parts.append(segment)

        return {
            "text": "".join(cleaned_parts),
            "dot_counts": dot_counts
        }

    def is_word_only(text):
        return bool(re.search(r'[a-zA-Z]', text))

    # Rest of the document processing code remains the same
    table_idx = 0
    para_idx = 0

    for block in doc.element.body:
        print(f'==================block {block.tag}')
        elements = []
        if block.tag.endswith('tbl'):
            if table_idx < len(doc.tables):
                table = doc.tables[table_idx]
                table_idx += 1

                for row in table.rows:
                    for cell in row.cells:
                        for para in cell.paragraphs:
                            para.alignment = WD_ALIGN_PARAGRAPH.DISTRIBUTE
                            for run in para.runs:
                                text = run.text or ""
                                if not text or not text.strip() or not is_word_only(text):
                                    continue

                                cleaned = clean_text(text.strip())
                                elements.append({
                                    'content': cleaned['text'].lower() if text.isupper() else cleaned['text'],
                                    'run': run,
                                    'type': 'table',
                                    'dot_counts': cleaned['dot_counts']
                                })

        elif block.tag.endswith('p'):
            if para_idx < len(doc.paragraphs):
                paragraph = doc.paragraphs[para_idx]
                para_idx += 1

                for run in paragraph.runs:
                    text = run.text
                    if not text or not text.strip() or not is_word_only(text):
                        continue

                    cleaned = clean_text(text.strip())
                    elements.append({
                        'content': cleaned['text'].lower() if text.isupper() else cleaned['text'],
                        'run': run,
                        'type': 'para',
                        'dot_counts': cleaned['dot_counts']
                    })

        if elements:
            batch_translate(elements, src_lang, tgt_lang, en_indic_model, en_indic_tokenizer, ip)

            for elem in elements:
                translated = elem.get('content')
                run = elem.get('run')
                if translated and run:
                    translated = translated.strip()
                    if translated:
                        if "[000]" in translated:
                            translated = restore_dots(translated, elem['dot_counts'])
                        run.text = translated
                        run.font.name = FONT_NAME
                        run.font.italic = False

    doc.save("/Users/shreesharya/Documents/Development/Translator/resource/output/translated.docx")