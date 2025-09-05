# azure_embeddings.py
import os
import openai

# Azure OpenAI config
openai.api_type = "azure"
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_version = "2024-02-01"

# Default to v3 small for cost/accuracy balance
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")

async def get_azure_embedding(text: str):
    """Get embedding vector from Azure OpenAI."""
    if not text.strip():
        return None
    response = openai.Embedding.create(
        input=text,
        engine=DEPLOYMENT_NAME
    )
    return response["data"][0]["embedding"]
