from src.utils.chunks import chunk_text
from src.utils.getfile import get_text_from_file
from src.utils.embedding import embedding

if __name__ == "__main__":
    remote_pdf_url = "https://arxiv.org/pdf/1709.00666.pdf"
    pdf_filename = "ch02-downloaded.pdf"

    chunk_size = 500
    overlap = 40

    text = get_text_from_file(remote_pdf_url, pdf_filename)

    chunks = chunk_text(text, chunk_size, overlap)

    embeddings = embedding(chunks)

    print(len(chunks))
