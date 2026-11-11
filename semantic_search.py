
import json
import math
from pathlib import Path

from google import genai
from google.genai import types

PROJECT_ID = "gcp-ai-rag-project"
LOCATION = "us-central1"
MODEL = "gemini-embedding-001"

DATA_FILE = Path("data/document_embeddings.json")

client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION,
)

def embed_question(question):
    response = client.models.embed_content(
        model=MODEL,
        contents=[question],
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY"
        ),
    )
    return response.embeddings[0].values

def cosine_similarity(a, b):
    if len(a) != len(b):
        raise ValueError("Embedding dimensions do not match")

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)

def search(question, top_k=2):
    records = json.loads(
        DATA_FILE.read_text(encoding="utf-8")
    )

    query_vector = embed_question(question)
    results = []

    for record in records:
        score = cosine_similarity(
            query_vector,
            record["embedding"]
        )

        results.append({
            "score": score,
            "text": record["text"],
            "source": record["source"],
            "page": record["page"],
        })

    results.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return results[:top_k]

if __name__ == "__main__":
    question = input("Ask a question: ").strip()

    if not question:
        raise ValueError("Please enter a question")

    matches = search(question)

    for index, match in enumerate(matches, start=1):
        print(f"\nResult {index}")
        print(f"Similarity: {match['score']:.4f}")
        print(f"Source: {match['source']}")
        print(f"Page: {match['page']}")
        print(f"Text:\n{match['text']}")
        print("-" * 50)
