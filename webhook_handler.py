import json
import httpx
import logging
from github_utils import get_valid_github_token
from issue_similarity import find_similar_issues

logger = logging.getLogger("gitai")

async def process_github_webhook(request, ai_service, logger):
    """Process GitHub webhook events."""
    payload = await request.json()  # Ensure the request object is used to parse JSON

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
        "creator_type": user.get("type"),  # User or Bot
        "labels": [label.get("name") for label in issue.get("labels", [])],
        "created_at": issue.get("created_at"),
        "language": repo.get("language"),
        "repo_topics": repo.get("topics", [])
    }

    try:
        resolution_draft = await ai_service.generate_resolution_draft(issue_data)

        github_token = get_valid_github_token()
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

        # Post draft to GitHub
        comment_url = f"https://api.github.com/repos/{issue_data['repo_name']}/issues/{issue_data['issue_number']}/comments"
        headers = {
            "Authorization": f"Bearer {github_token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "git-ai-bot"
        }
        comment_body = {
            "body": f"**[Draft Resolution Steps by Git.AI]**\n\n{resolution_draft}{similar_section}"
        }
        async with httpx.AsyncClient() as client:
            resp = await client.post(comment_url, headers=headers, json=comment_body)
            if resp.status_code == 201:
                logger.info("Draft comment posted to GitHub issue.")
            else:
                logger.warning(f"Failed to post comment: {resp.status_code} {resp.text}")

        return {
            "status": "Success",
            "issue_processed": {
                "repo": issue_data["repo_name"],
                "issue_number": issue_data["issue_number"],
                "title": issue_data["issue_title"]
            },
            "resolution_draft": resolution_draft,
            "similar_issues": [ {"title": t, "url": u} for u, t in similar_issues ]
        }

    except Exception as e:
        logger.error(f"Error: {e}")
        return {"status": "Error", "message": str(e), "issue_data": issue_data}
