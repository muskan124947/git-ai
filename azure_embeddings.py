# azure_embeddings.py
import os
from openai import AzureOpenAI

# Azure OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2023-12-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

# Default to v3 small for cost/accuracy balance
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")

async def get_azure_embedding(text: str):
    """Get embedding vector from Azure OpenAI."""
    if not text.strip():
        return None
    
    # Truncate text to stay within token limits
    # Roughly 4 characters per token, so ~30,000 chars for 8192 tokens
    # Being conservative, we'll use 20,000 characters max
    max_chars = 20000
    if len(text) > max_chars:
        text = text[:max_chars]
        print(f"Warning: Text truncated from {len(text)} to {max_chars} characters for embedding")
    
    try:
        response = client.embeddings.create(
            input=text,
            model=DEPLOYMENT_NAME
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Embedding error: {e}")
        # If still too long, try with even shorter text
        if "maximum context length" in str(e) and len(text) > 5000:
            print("Retrying with shorter text...")
            shortened_text = text[:5000]
            try:
                response = client.embeddings.create(
                    input=shortened_text,
                    model=DEPLOYMENT_NAME
                )
                return response.data[0].embedding
            except Exception as e2:
                print(f"Second embedding attempt failed: {e2}")
                return None
        return None
