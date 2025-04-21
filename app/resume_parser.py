import pdfminer.high_level

def extract_text_from_resume(file_path):
    return pdfminer.high_level.extract_text(file_path)
