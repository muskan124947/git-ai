# ai_summary.py
import os
from openai import AzureOpenAI

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2023-12-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

async def generate_summary(issue_data):
    """Generate a short AI summary of the issue."""
    issue_text = f"Title: {issue_data['issue_title']}\n\nBody: {issue_data['issue_body']}"
    prompt = f"""
    Summarize the following GitHub issue in 2-3 concise sentences for a maintainer:

    {issue_text}
    """
    response = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful GitHub issue summarizer."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100,
        temperature=0.5
    )
    return response.choices[0].message.content.strip()
