# main.py
from fastapi import FastAPI, Request
import json

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Git.AI backend is running!"}

@app.post("/webhook")
async def github_webhook(request: Request):
    payload = await request.json()

    print("\n New GitHub Webhook Event Received:")
    
    # Extract issue and repository info
    issue = payload.get("issue", {})
    repo = payload.get("repository", {})

    data = {
        "repo_name": repo.get("full_name"),
        "issue_number": issue.get("number"),
        "issue_title": issue.get("title"),
        "issue_body": issue.get("body"),
        "issue_url": issue.get("html_url"),
        "creator": issue.get("user", {}).get("login"),
    }

    print(json.dumps(data, indent=2))
    return {"status": "Webhook received"}
