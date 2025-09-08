# ai_priority.py
import os
from openai import AzureOpenAI

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2023-12-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

async def generate_priority(issue_data):
    """Classify issue priority as High, Medium, or Low."""
    issue_text = f"Title: {issue_data['issue_title']}\n\nBody: {issue_data['issue_body']}"
    prompt = f"""
    Based on the following GitHub issue, classify its priority for maintainers as High, Medium, or Low.
    Provide only one word: High, Medium, or Low.

    {issue_text}
    """
    response = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[
            {"role": "system", "content": "You are an issue triage assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=10,
        temperature=0
    )
    return response.choices[0].message.content.strip()
