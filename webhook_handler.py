# webhook_handler.py
import json
import httpx
import logging
from github_utils import get_valid_github_token, get_repo_labels, add_label_to_issue
from issue_similarity import find_similar_issues
from labeler import predict_label
from ai_summary import generate_summary
from ai_priority import generate_priority

logger = logging.getLogger("gitai")

async def process_github_webhook(request, ai_service, logger):
    """Process GitHub webhook events."""
    payload = await request.json()

    if payload.get("action") != "opened":
        return {"status": "Ignored - not issue creation"}

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
        "creator_type": user.get("type"),
        "labels": [label.get("name") for label in issue.get("labels", [])],
        "created_at": issue.get("created_at"),
        "language": repo.get("language"),
        "repo_topics": repo.get("topics", [])
    }

    try:
        github_token = get_valid_github_token()

        # 1️⃣ Generate AI draft (still using ai_service)
        resolution_draft = await ai_service.generate_resolution_draft(issue_data)

        # 2️⃣ New features: summary + priority
        summary = await generate_summary(issue_data)
        priority = await generate_priority(issue_data)

        # 3️⃣ Find similar issues
        similar_issues = await find_similar_issues(
            issue_data["repo_name"],
            issue_data["issue_number"],
            issue_data["issue_title"],
            issue_data["issue_body"],
            github_token
        )

        similar_section = ""
        if similar_issues:
            similar_section = "\n\n**Similar Issues (semantic match):**\n"
            for issue in similar_issues:
                similar_section += f"- [{issue['title']}]({issue['url']}) (score: {issue['score']})\n"
        else:
            similar_section = "\n\n**Similar Issues:**\n_None found_"

        # 4️⃣ Predict and apply label
        predicted_label = None
        available_labels = await get_repo_labels(issue_data["repo_name"], github_token)
        if available_labels:
            predicted_label = await predict_label(
                issue_data["issue_title"],
                issue_data["issue_body"],
                available_labels
            )
            if predicted_label:
                await add_label_to_issue(
                    issue_data["repo_name"],
                    issue_data["issue_number"],
                    predicted_label,
                    github_token
                )

        # 5️⃣ Post AI comment
        comment_url = f"https://api.github.com/repos/{issue_data['repo_name']}/issues/{issue_data['issue_number']}/comments"
        headers = {
            "Authorization": f"Bearer {github_token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "git-ai-bot"
        }

        label_section = f"\n\n**Auto-applied Label:** `{predicted_label}`" if predicted_label else ""

        comment_body = {
            "body": f"""
**[AI Summary]**
{summary}

**[Draft Resolution Steps by Git.AI]**
{resolution_draft}

**[Triage Priority]**
{priority}{label_section}{similar_section}
"""
        }

        async with httpx.AsyncClient() as client:
            resp = await client.post(comment_url, headers=headers, json=comment_body)
            if resp.status_code == 201:
                logger.info("AI summary, draft, label, and priority posted to GitHub issue.")
            else:
                logger.warning(f"Failed to post comment: {resp.status_code} {resp.text}")

        return {
            "status": "Success",
            "issue_processed": {
                "repo": issue_data["repo_name"],
                "issue_number": issue_data["issue_number"],
                "title": issue_data["issue_title"]
            },
            "summary": summary,
            "resolution_draft": resolution_draft,
            "priority": priority,
            "applied_label": predicted_label,
            "similar_issues": similar_issues
        }

    except Exception as e:
        logger.error(f"Error: {e}")
        return {"status": "Error", "message": str(e), "issue_data": issue_data}
