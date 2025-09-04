from sentence_transformers import SentenceTransformer, util
from github_utils import get_past_issues

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def compute_issue_embedding(title, body):
    text = (title or "") + " " + (body or "")
    return embedding_model.encode(text, convert_to_tensor=True)

async def find_similar_issues(repo_full_name, new_issue_number, new_title, new_body, github_token, max_results=3):
    # Step 1: Fetch past issues
    past_issues = await get_past_issues(repo_full_name, github_token, max_issues=50)
    if not past_issues:
        return []

    # Step 2: Compute embedding for new issue
    new_embedding = compute_issue_embedding(new_title, new_body)

    # Step 3: Compute similarity against past issues
    similarities = []

    for issue in past_issues:
        issue_title = issue.get("title", "")
        issue_body = issue.get("body", "")
        issue_embedding = compute_issue_embedding(issue_title, issue_body)

        sim_score = util.cos_sim(new_embedding, issue_embedding).item()
        similarities.append((sim_score, issue))

    # Step 4: Sort by similarity
    similarities.sort(key=lambda x: x[0], reverse=True)

    # Step 5: Pick top-k similar issues (excluding current issue)
    similar = []
    for score, issue in similarities:
        if issue["number"] == new_issue_number:
            continue
        if score < 0.55:  # similarity threshold
            break
        similar.append({
            "url": issue["html_url"],
            "title": issue["title"],
            "score": round(score, 3)
        })
        if len(similar) >= max_results:
            break

    return similar
