# main.py
from fastapi import FastAPI, Request
import json
from ai_service import AIService
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()
ai_service = AIService()

@app.get("/")
def root():
    return {"message": "Git.AI backend is running!"}

@app.post("/webhook")
async def github_webhook(request: Request):
    payload = await request.json()
    
    # Only process issue creation events
    if payload.get("action") != "opened":
        return {"status": "Event ignored - not an issue creation"}

    print("\n🔔 New GitHub Issue Created:")
    
    # Extract comprehensive issue and repository info
    issue = payload.get("issue", {})
    repo = payload.get("repository", {})
    user = issue.get("user", {})
    
    issue_data = {
        "repo_name": repo.get("full_name"),
        "repo_description": repo.get("description"),
        "issue_number": issue.get("number"),
        "issue_title": issue.get("title"),
        "issue_body": issue.get("body"),
        "issue_url": issue.get("html_url"),
        "creator": user.get("login"),
        "creator_type": user.get("type"),  # User or Bot
        "labels": [label.get("name") for label in issue.get("labels", [])],
        "created_at": issue.get("created_at"),
        "language": repo.get("language"),
        "repo_topics": repo.get("topics", [])
    }

    print(json.dumps(issue_data, indent=2))
    
    try:
        # Generate AI resolution steps
        print("\n🤖 Generating AI resolution steps...")
        resolution_draft = await ai_service.generate_resolution_draft(issue_data)
        
        print(f"\n✅ Generated Resolution Draft:")
        print("=" * 60)
        print(resolution_draft)  # Show full resolution instead of truncated
        print("=" * 60)
        
        return {
            "status": "Success",
            "issue_processed": {
                "repo": issue_data["repo_name"],
                "issue_number": issue_data["issue_number"],
                "title": issue_data["issue_title"]
            },
            "resolution_draft": resolution_draft
        }
        
    except Exception as e:
        print(f"❌ Error generating resolution: {str(e)}")
        return {
            "status": "Error", 
            "message": f"Failed to generate resolution: {str(e)}",
            "issue_data": issue_data
        }
