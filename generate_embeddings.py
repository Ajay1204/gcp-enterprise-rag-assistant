
import json
from pathlib import Path

from google import genai
from google.genai import types

from chunk_documents import load_pdf, chunk_text

PROJECT_ID = "gcp-ai-rag-project"
LOCATION = "us-central1"
MODEL = "gemini-embedding-001"
OUTPUT_FILE = Path("data/document_embeddings.json")

client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION,
)

def generate_embedding(text):
    response = client.models.embed_content(
        model=MODEL,
        contents=[text],
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT"
        ),
    )
    return response.embeddings[0].values

def main():
    pages = load_pdf()
    records = []

    for page in pages:
        chunks = chunk_text(page["text"])

        for index, chunk in enumerate(chunks, start=1):
            if not chunk.strip():
                continue

            print(f"Embedding page {page['page']}, chunk {index}")

            vector = generate_embedding(chunk)

            records.append({
                "id": f"page-{page['page']}-chunk-{index}",
                "source": "Employee_Handbook_2026_Sample.pdf",
                "page": page["page"],
                "text": chunk,
                "embedding": vector,
            })

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_FILE.write_text(
        json.dumps(records, indent=2),
        encoding="utf-8"
    )

    print(f"\nSaved {len(records)} embeddings")
    print(f"File: {OUTPUT_FILE}")
    print(f"Dimensions: {len(records[0]['embedding'])}")

if __name__ == "__main__":
    main()
