# Enterprise RAG Assistant on Google Cloud

A Python-based document retrieval system using Google Cloud Vertex AI, Cloud Storage, and FastAPI.

## Overview

This project demonstrates how to build a cloud-integrated semantic search backend for enterprise documents.

Users can submit natural-language questions through a REST API and retrieve relevant document passages with similarity scores and source references.

## Architecture

```text
Google Cloud Storage
        |
        v
PDF Extraction (pypdf)
        |
        v
Text Chunking
        |
        v
Vertex AI Embeddings
        |
        v
JSON Vector Storage
        |
        v
Semantic Search (Cosine Similarity)
        |
        v
FastAPI REST API
        |
        v
JSON Results with Source References
```

## Technology Stack

- **Cloud:** Google Cloud Platform (GCP)
- **Storage:** Google Cloud Storage
- **AI:** Vertex AI Embeddings
- **Backend:** Python, FastAPI
- **Vector Retrieval:** Cosine Similarity
- **Document Processing:** pypdf
- **API Documentation:** Swagger UI
- **Version Control:** Git and GitHub

## Features

- PDF document extraction
- Text chunking with overlap
- Vertex AI embedding generation
- Semantic similarity search
- REST API for document retrieval
- Similarity scores and source references
- Interactive API testing using Swagger UI

## API Endpoints

### GET /health

Checks API availability.

### POST /search

Example request:

```json
{
  "question": "How many vacation days do employees receive?",
  "top_k": 2
}
```

Returns relevant passages, similarity scores, and document source information.

## Getting Started

1. Clone the repository.
2. Create a Python virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Configure Google Cloud authentication and Vertex AI access.
5. Upload your PDF to a configured Cloud Storage bucket.
6. Generate document embeddings.
7. Start the API:

```bash
uvicorn app:app --host 0.0.0.0 --port 8080
```

8. Open `/docs` to test the endpoints.

## Current Status

- PDF extraction: Complete
- Document chunking: Complete
- Vertex AI embeddings: Complete
- Semantic search: Complete
- FastAPI search endpoint: Tested successfully
- LLM-generated answers: In progress
- Production deployment: Planned

## Future Improvements

- Integrate an accessible Gemini model for grounded answer generation
- Add a managed vector database
- Implement authentication and automated tests
- Containerize with Docker
- Deploy to Google Cloud Run
- Add monitoring and evaluation metrics

## Sample Data

This project uses a fictional employee handbook for demonstration purposes. Its policies are not actual employment guidance.

## Purpose

Built as a hands-on AI engineering portfolio project demonstrating cloud AI integration, semantic retrieval, Python backend development, and REST API design.
