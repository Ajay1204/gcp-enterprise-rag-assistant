from google import genai
from google.genai import types

# 1. Connect to Vertex AI
client = genai.Client(
    vertexai=True,
    project="gcp-ai-rag-project",
    location="us-central1"
)

# 2. Text from our Employee Handbook
text = "Employees receive 20 paid vacation days per year."

# 3. Convert the text into an embedding
response = client.models.embed_content(
    model="gemini-embedding-001",
    contents=[text],
    config=types.EmbedContentConfig(
        task_type="RETRIEVAL_DOCUMENT"
    )
)

# 4. Extract the numerical vector
vector = response.embeddings[0].values

# 5. Display the result
print("Original text:", text)
print("Number of dimensions:", len(vector))
print("First 10 values:", vector[:10])
