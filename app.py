from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from semantic_search import search

app = FastAPI(
    title="Enterprise RAG Assistant",
    description="Document semantic search using Google Vertex AI",
    version="1.0.0"
)


class SearchRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=2, ge=1, le=10)


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/search")
def search_documents(request: SearchRequest):
    try:
        results = search(request.question, top_k=request.top_k)

        return {
            "question": request.question,
            "results": results
        }

    except Exception as error:
        print(f"Search failed: {error}")
        raise HTTPException(
            status_code=500,
            detail="Document search failed"
        )
