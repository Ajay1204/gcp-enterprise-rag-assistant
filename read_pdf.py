import io
from google.cloud import storage
from pypdf import PdfReader

PROJECT_ID = "gcp-ai-rag-project"
BUCKET_NAME = "gcp-ai-rag-documents-2026"
PDF_NAME = "Employee_Handbook_2026_Sample.pdf"

# Connect to Google Cloud Storage
client = storage.Client(project=PROJECT_ID)

# Find the PDF
bucket = client.bucket(BUCKET_NAME)
blob = bucket.blob(PDF_NAME)

# Download PDF
pdf_bytes = blob.download_as_bytes()

# Read PDF
reader = PdfReader(io.BytesIO(pdf_bytes))

for page_number, page in enumerate(reader.pages, start=1):
    print(f"\n--- Page {page_number} ---")
    print(page.extract_text() or "No text found")
