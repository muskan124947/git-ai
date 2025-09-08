# labeler.py
import os
import logging
from openai import AzureOpenAI
from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger("gitai")

# --- Azure OpenAI Config ---
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2023-12-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

# --- Local Embedding Model ---
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

async def predict_label_with_openai(title, body, available_labels):
    """Try Azure OpenAI classification first."""
    issue_text = f"Title: {title}\n\nBody: {body}"

    prompt = f"""
    You are an issue triage assistant.
    Given this GitHub issue, assign the most relevant label from the list below.

    Issue:
    {issue_text}

    Labels: {", ".join(available_labels)}

    Return only the best label name, nothing else.
    """

    try:
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful GitHub issue triage assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=20,
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"OpenAI label prediction failed: {e}")
        return None


def predict_label_with_embeddings(title, body, available_labels):
    """Fallback: use semantic similarity with SentenceTransformers."""
    text = (title or "") + " " + (body or "")
    issue_embedding = embedding_model.encode(text, convert_to_tensor=True)

    label_embeddings = embedding_model.encode(available_labels, convert_to_tensor=True)
    similarities = util.cos_sim(issue_embedding, label_embeddings)

    best_idx = int(similarities.argmax())
    return available_labels[best_idx]


async def predict_label(title, body, available_labels):
    """Hybrid label predictor with fallback."""
    label = await predict_label_with_openai(title, body, available_labels)
    if label and label in available_labels:
        return label
    else:
        return predict_label_with_embeddings(title, body, available_labels)
