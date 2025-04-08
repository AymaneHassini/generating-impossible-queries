import PyPDF2

def extract_text_from_pdf(pdf_path):
  try:
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        extracted_text = ""
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                extracted_text += text
        return extracted_text 
  except Exception as e:
    print(f"Failed to process document from {pdf_path}: {e}")
    return None