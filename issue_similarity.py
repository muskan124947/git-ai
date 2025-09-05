# issue_similarity.py
import numpy as np
from github_utils import get_past_issues
from azure_embeddings import get_azure_embedding

async def find_similar_issues(repo_full_name, new_issue_number, new_title, new_body, github_token, max_results=3):
    # Step 1: Fetch past issues
    past_issues = await get_past_issues(repo_full_name, github_token, max_issues=50)
    if not past_issues:
        return []

    # Step 2: Embedding for new issue (Azure)
    new_text = (new_title or "") + " " + (new_body or "")
    new_embedding = await get_azure_embedding(new_text)
    if new_embedding is None:
        return []

    similarities = []
    for issue in past_issues:
        issue_text = (issue.get("title", "") or "") + " " + (issue.get("body", "") or "")
        issue_embedding = await get_azure_embedding(issue_text)
        if issue_embedding is None:
            continue

        # Cosine similarity
        sim_score = float(
            np.dot(new_embedding, issue_embedding) /
            (np.linalg.norm(new_embedding) * np.linalg.norm(issue_embedding))
        )
        similarities.append((sim_score, issue))

    # Step 3: Sort & filter
    similarities.sort(key=lambda x: x[0], reverse=True)

    similar = []
    for score, issue in similarities:
        if issue["number"] == new_issue_number:
            continue
        if score < 0.75:  # similarity threshold
            break
        similar.append({
            "url": issue["html_url"],
            "title": issue["title"],
            "score": round(score, 3)
        })
        if len(similar) >= max_results:
            break

    return similar
