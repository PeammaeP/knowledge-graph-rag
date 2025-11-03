import requests
import pdfplumber

def get_text_from_file(remote_pdf_url, pdf_filename: str):
    response = requests.get(remote_pdf_url)

    text = ""

    if response.status_code == 200:
        with open(pdf_filename, "wb") as pdf_file:
            pdf_file.write(response.content)

    else:
        print("Failed to download the PDF. Status code:", response.status_code)

    with pdfplumber.open(pdf_filename) as pdf:
        for page in pdf.pages:
            text += page.extract_text()

    return text
