# ai_priority.py
import os
import openai

openai.api_type = "azure"
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_version = "2024-02-01"

DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

async def generate_priority(issue_data):
    """Classify issue priority as High, Medium, or Low."""
    issue_text = f"Title: {issue_data['issue_title']}\n\nBody: {issue_data['issue_body']}"
    prompt = f"""
    Based on the following GitHub issue, classify its priority for maintainers as High, Medium, or Low.
    Provide only one word: High, Medium, or Low.

    {issue_text}
    """
    response = openai.ChatCompletion.create(
        engine=DEPLOYMENT_NAME,
        messages=[
            {"role": "system", "content": "You are an issue triage assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=10,
        temperature=0
    )
    return response["choices"][0]["message"]["content"].strip()
