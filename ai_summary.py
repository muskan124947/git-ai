# ai_summary.py
import os
import openai

openai.api_type = "azure"
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_version = "2024-02-01"

DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

async def generate_summary(issue_data):
    """Generate a short AI summary of the issue."""
    issue_text = f"Title: {issue_data['issue_title']}\n\nBody: {issue_data['issue_body']}"
    prompt = f"""
    Summarize the following GitHub issue in 2-3 concise sentences for a maintainer:

    {issue_text}
    """
    response = openai.ChatCompletion.create(
        engine=DEPLOYMENT_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful GitHub issue summarizer."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100,
        temperature=0.5
    )
    return response["choices"][0]["message"]["content"].strip()
