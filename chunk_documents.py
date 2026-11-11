
import io
from google.cloud import storage
from pypdf import PdfReader

PROJECT_ID = "gcp-ai-rag-project"
BUCKET_NAME = "gcp-ai-rag-documents-2026"
PDF_NAME = "Employee_Handbook_2026_Sample.pdf"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

def load_pdf():
    client = storage.Client(project=PROJECT_ID)
    blob = client.bucket(BUCKET_NAME).blob(PDF_NAME)
    pdf_bytes = blob.download_as_bytes()

    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = []

    for page_number, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({
                "page": page_number,
                "text": text.strip()
            })

    return pages

def chunk_text(text, size=500, overlap=100):
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("Invalid chunk size or overlap")

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])

        if end == len(text):
            break

        start = end - overlap

    return chunks

if __name__ == "__main__":
    pages = load_pdf()

    for page in pages:
        chunks = chunk_text(
            page["text"],
            CHUNK_SIZE,
            CHUNK_OVERLAP
        )

        for index, chunk in enumerate(chunks, 1):
            print(f"\nPage {page['page']} - Chunk {index}")
            print(chunk)
            print("-" * 50)
