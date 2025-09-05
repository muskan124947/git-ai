import httpx
import logging
from github import GithubIntegration
from config import APP_ID, INSTALLATION_ID, PRIVATE_KEY_PATH

logger = logging.getLogger("gitai")

def get_github_app_token():
    """Generate GitHub App token"""
    with open(PRIVATE_KEY_PATH, "r") as key_file:
        private_key = key_file.read()
    git_integration = GithubIntegration(APP_ID, private_key)
    return git_integration.get_access_token(INSTALLATION_ID).token

# Generate token at runtime (before each request, to avoid expiry)
def get_valid_github_token():
    return get_github_app_token()

async def get_past_issues(repo_full_name, github_token, max_issues=50):
    """Fetch issues from GitHub repo"""
    issues_url = f"https://api.github.com/repos/{repo_full_name}/issues"
    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "git-ai-bot"
    }
    params = {"state": "all", "per_page": max_issues}
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(issues_url, headers=headers, params=params)
            if resp.status_code == 200:
                return [issue for issue in resp.json() if "pull_request" not in issue]
            else:
                logger.warning(f"Failed to fetch past issues: {resp.status_code} {resp.text}")
    except Exception as e:
        logger.error(f"Exception fetching past issues: {e}")
    return []

# --- ADD BELOW TO github_utils.py ---
async def get_repo_labels(repo_full_name, github_token):
    """Fetch all available labels in a repo."""
    url = f"https://api.github.com/repos/{repo_full_name}/labels"
    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "git-ai-bot"
    }
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, headers=headers)
        if resp.status_code == 200:
            return [label["name"] for label in resp.json()]
        return []


async def add_label_to_issue(repo_full_name, issue_number, labels, github_token):
    """Apply labels to a GitHub issue."""
    url = f"https://api.github.com/repos/{repo_full_name}/issues/{issue_number}/labels"
    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "git-ai-bot"
    }
    data = {"labels": labels if isinstance(labels, list) else [labels]}
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, headers=headers, json=data)
        return resp.status_code in (200, 201)

