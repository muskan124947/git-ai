from openai import AzureOpenAI
import openai
import os
import requests
import base64
from typing import Dict, Any, List, Optional
import json

class AIService:
    def __init__(self):
        # Enhanced Azure OpenAI configuration (supports both old and new clients)
        try:
            # Try new v1.0+ format first
            self.client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version="2023-12-01-preview",
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
            )
            self.use_new_client = True
        except Exception as e:
            # Fallback to legacy client with new syntax
            print(f"Creating legacy client: {e}")
            self.legacy_client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version="2023-12-01-preview",
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
            )
            self.use_new_client = False
        
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-35-turbo")
        
        # GitHub API configuration for enhanced context
        self.github_token = os.getenv("GITHUB_TOKEN")  # Optional: for private repos
        
        # Tunable parameters for different issue types - Enhanced for longer responses
        self.model_parameters = {
            "bug": {
                "temperature": 0.3,  # More deterministic for bugs
                "max_tokens": 8000,  # Increased from original
                "top_p": 0.8
            },
            "feature": {
                "temperature": 0.7,  # More creative for features
                "max_tokens": 8000,  # Increased from original
                "top_p": 0.9
            },
            "documentation": {
                "temperature": 0.4,  # Balanced for docs
                "max_tokens": 6000,  # Increased from original
                "top_p": 0.85
            },
            "question": {
                "temperature": 0.5,  # Balanced for questions
                "max_tokens": 6000,  # Increased from original
                "top_p": 0.85
            },
            "default": {
                "temperature": 0.6,
                "max_tokens": 8000,  # Increased from original 4000
                "top_p": 0.85
            }
        }
        
        if not os.getenv("AZURE_OPENAI_API_KEY") or not os.getenv("AZURE_OPENAI_ENDPOINT"):
            print("⚠️  Warning: Azure OpenAI credentials not found in environment variables")
            print("Please set: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT_NAME")

    async def generate_resolution_draft(self, issue_data: Dict[str, Any]) -> str:
        """Enhanced resolution draft generation with repository context"""
        
        # Collect enhanced repository context
        repo_context = await self._collect_repository_context(issue_data)
        
        # Detect issue type for parameter tuning
        issue_type = self._detect_issue_type(issue_data)
        
        # Get tuned parameters
        params = self.model_parameters.get(issue_type, self.model_parameters["default"])
        
        # Build enhanced prompt with repository context
        prompt = self._build_enhanced_prompt(issue_data, repo_context, issue_type)
        
        try:
            if self.use_new_client:
                # Use new v1.0+ client
                response = self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=[
                        {
                            "role": "system", 
                            "content": self._get_system_prompt(issue_type)
                        },
                        {
                            "role": "user", 
                            "content": prompt
                        }
                    ],
                    max_tokens=params["max_tokens"],
                    temperature=params["temperature"],
                    top_p=params["top_p"],
                    frequency_penalty=0.1,  # Reduce repetition
                    presence_penalty=0.1    # Encourage diverse topics
                )
                return response.choices[0].message.content.strip()
            else:
                # Use legacy client with new v1.0+ syntax
                response = self.legacy_client.chat.completions.create(
                    model=self.deployment_name,
                    messages=[
                        {
                            "role": "system", 
                            "content": self._get_system_prompt(issue_type)
                        },
                        {
                            "role": "user", 
                            "content": prompt
                        }
                    ],
                    max_tokens=params["max_tokens"],
                    temperature=params["temperature"],
                    top_p=params["top_p"],
                    frequency_penalty=0.1,
                    presence_penalty=0.1
                )
                return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Azure OpenAI API error: {str(e)}")
            # Fallback to enhanced template-based response
            return self._generate_enhanced_fallback_response(issue_data, repo_context)
    
    async def _collect_repository_context(self, issue_data: Dict[str, Any]) -> Dict[str, Any]:
        """Repository context collection with relevant data from reference repository"""
        
        # Always target the reference repository for enhanced context
        target_repo = "microsoft/mssql-jdbc"
        original_repo = issue_data.get('repo_name', '')
        
        context = {
            "reference_commits": [],
            "reference_open_issues": [],
            "reference_closed_issues": [],
            "related_issues": [],
            "readme_content": None,
            "original_repo_context": {},
            "is_reference_repo": original_repo == target_repo
        }
        
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "Git-AI-Bot"
        }
        
        # Use the GitHub token for better rate limits
        if self.github_token:
            headers["Authorization"] = f"token {self.github_token}"
        
        try:
            print(f"🔍 Collecting repository context from: {target_repo}")
            print(f"📋 Original issue repository: {original_repo}")
            
            # Get reference repository recent commits (filter for relevant ones)
            print("📥 Fetching reference repository recent commits...")
            reference_commits = await self._get_filtered_commits(target_repo, headers, issue_data, max_commits=30)
            context["reference_commits"] = reference_commits

            # Get reference repository open issues (filter for relevant ones)
            print("📥 Fetching reference repository open issues...")
            reference_open_issues = await self._get_filtered_issues(target_repo, headers, "open", issue_data, max_issues=50)
            context["reference_open_issues"] = reference_open_issues

            # Get reference repository closed issues (filter for relevant ones)
            print("📥 Fetching reference repository closed issues...")
            reference_closed_issues = await self._get_filtered_issues(target_repo, headers, "closed", issue_data, max_issues=100)
            context["reference_closed_issues"] = reference_closed_issues

            # Find related issues
            print("🔍 Finding related issues...")
            all_issues = reference_open_issues + reference_closed_issues
            related_issues = self._find_related_issues(all_issues, issue_data)
            context["related_issues"] = related_issues
            
            # Get reference repository README for context
            print("📄 Fetching reference repository README...")
            readme = await self._get_readme_content(target_repo, headers)
            context["readme_content"] = readme
            
            # If the original issue is NOT from reference repo, get minimal context from original repo
            if not context["is_reference_repo"]:
                print(f"📄 Getting minimal context from original repo: {original_repo}")
                original_context = await self._get_minimal_original_context(original_repo, headers, issue_data)
                context["original_repo_context"] = original_context
                
        except Exception as e:
            print(f"Error collecting repository context: {str(e)}")
        
        return context

    async def _get_filtered_commits(self, repo_name: str, headers: dict, issue_data: Dict, max_commits: int = 30) -> List[Dict]:
        """Get commits filtered for relevance using AI-powered analysis"""
        commits = []
        
        try:
            commits_url = f"https://api.github.com/repos/{repo_name}/commits"
            params = {"per_page": max_commits}
            
            response = requests.get(commits_url, headers=headers, params=params, timeout=15)
            if response.status_code == 200:
                commit_data = response.json()
                
                # Prepare data for AI filtering
                issue_context = f"Issue: {issue_data.get('issue_title', '')}\n{issue_data.get('issue_body', '')}"
                
                # Process commits in batches for AI analysis
                batch_size = 10
                for i in range(0, len(commit_data), batch_size):
                    batch = commit_data[i:i+batch_size]
                    relevant_commits = await self._ai_filter_commits(batch, issue_context)
                    commits.extend(relevant_commits)
                    
                # Ensure we have at least 2 commits - add most recent if needed
                if len(commits) < 2 and len(commit_data) >= 2:
                    print("📝 Adding recent commits to ensure minimum of 2...")
                    for commit in commit_data[:4]:  # Check first 4 commits
                        if len(commits) >= 2:
                            break
                        # Add if not already included
                        if not any(c["sha"] == commit["sha"][:7] for c in commits):
                            commits.append({
                                "sha": commit["sha"][:7],
                                "message": commit["commit"]["message"][:250],
                                "author": commit["commit"]["author"]["name"],
                                "date": commit["commit"]["author"]["date"],
                                "url": commit["html_url"],
                                "ai_filtered": False  # Mark as backup commit
                            })
                    
            print(f"✅ AI-filtered {len(commits)} commits from {len(commit_data)} total (minimum 2 guaranteed)")
        except Exception as e:
            print(f"Error in AI commit filtering: {e}")
            # Fallback to simple filtering if AI fails
            commits = await self._fallback_commit_filter(repo_name, headers, issue_data, max_commits)
            
        return commits

    async def _ai_filter_commits(self, commits_batch: List[Dict], issue_context: str) -> List[Dict]:
        """Use OpenAI to intelligently filter commits for relevance"""
        try:
            # Prepare commit summaries for AI analysis
            commit_summaries = []
            for i, commit in enumerate(commits_batch):
                commit_summaries.append(f"{i}: {commit['commit']['message'][:200]}")
            
            commits_text = "\n".join(commit_summaries)
            
            # AI prompt for commit filtering
            filter_prompt = f"""You are an expert software engineer analyzing commit relevance.

ISSUE CONTEXT:
{issue_context}

COMMITS TO ANALYZE:
{commits_text}

TASK: Identify which commits are HIGHLY RELEVANT to solving the above issue.

CRITERIA for HIGH RELEVANCE:
- Commits that fix similar bugs or issues
- Commits that modify the same methods/classes mentioned in the issue
- Commits that address similar technical problems
- Commits that improve related functionality

CRITERIA for LOW RELEVANCE:
- General documentation updates
- Unrelated feature additions
- Different technical areas
- Cosmetic changes

OUTPUT FORMAT:
Return only the numbers (0-{len(commits_batch)-1}) of HIGHLY RELEVANT commits, separated by commas.
If no commits are highly relevant, return "NONE".

HIGHLY RELEVANT COMMIT NUMBERS:"""

            # Call OpenAI API
            if self.use_new_client:
                response = self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=[
                        {"role": "system", "content": "You are an expert code analyst specializing in commit relevance assessment."},
                        {"role": "user", "content": filter_prompt}
                    ],
                    temperature=0.1,  # Low temperature for consistent filtering
                    max_tokens=100
                )
                ai_response = response.choices[0].message.content.strip()
            else:
                response = self.legacy_client.chat.completions.create(
                    model=self.deployment_name,
                    messages=[
                        {"role": "system", "content": "You are an expert code analyst specializing in commit relevance assessment."},
                        {"role": "user", "content": filter_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=100
                )
                ai_response = response.choices[0].message.content.strip()
            
            # Parse AI response and extract relevant commits
            relevant_commits = []
            if ai_response.upper() != "NONE":
                try:
                    relevant_indices = [int(x.strip()) for x in ai_response.split(',') if x.strip().isdigit()]
                    for idx in relevant_indices:
                        if 0 <= idx < len(commits_batch):
                            commit = commits_batch[idx]
                            relevant_commits.append({
                                "sha": commit["sha"][:7],
                                "message": commit["commit"]["message"][:250],
                                "author": commit["commit"]["author"]["name"],
                                "date": commit["commit"]["author"]["date"],
                                "url": commit["html_url"],
                                "ai_filtered": True
                            })
                except ValueError as e:
                    print(f"Error parsing AI response: {e}")
            
            return relevant_commits
            
        except Exception as e:
            print(f"Error in AI commit filtering: {e}")
            return []

    async def _fallback_commit_filter(self, repo_name: str, headers: dict, issue_data: Dict, max_commits: int) -> List[Dict]:
        """Fallback filtering method if AI filtering fails"""
        commits = []
        
        try:
            commits_url = f"https://api.github.com/repos/{repo_name}/commits"
            params = {"per_page": max_commits}
            
            response = requests.get(commits_url, headers=headers, params=params, timeout=15)
            if response.status_code == 200:
                commit_data = response.json()
                
                # Extract key terms from issue
                issue_text = (issue_data.get('issue_title', '') + ' ' + issue_data.get('issue_body', '')).lower()
                key_terms = self._extract_key_terms(issue_text)
                
                for commit in commit_data[:10]:  # Limit to most recent 10 for fallback
                    commit_message = commit["commit"]["message"].lower()
                    
                    # Simple relevance check
                    if any(term in commit_message for term in key_terms):
                        commits.append({
                            "sha": commit["sha"][:7],
                            "message": commit["commit"]["message"][:250],
                            "author": commit["commit"]["author"]["name"],
                            "date": commit["commit"]["author"]["date"],
                            "url": commit["html_url"],
                            "ai_filtered": False
                        })
                    
        except Exception as e:
            print(f"Error in fallback filtering: {e}")
            
        return commits

    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms for fallback filtering"""
        import re
        
        # Extract method names, class names, and key technical terms
        terms = []
        
        # Method names (camelCase with parentheses)
        method_matches = re.findall(r'\b[a-z][a-zA-Z]*\(\)', text)
        terms.extend([match.replace('()', '') for match in method_matches])
        
        # Common technical terms
        tech_terms = ['fix', 'bug', 'error', 'issue', 'batch', 'statement', 'result', 'connection', 'jdbc', 'sql']
        terms.extend([term for term in tech_terms if term in text])
        
        return list(set(terms))

    async def _get_filtered_issues(self, repo_name: str, headers: dict, state: str, issue_data: Dict, max_issues: int = 50) -> List[Dict]:
        """Get issues filtered for relevance"""
        issues = []
        
        # General keywords and patterns for relevance
        relevant_keywords = ['bug', 'fix', 'issue', 'error', 'connection', 'timeout', 'performance', 'exception', 'feature', 'enhancement', 'problem', 'help']
        
        try:
            issues_url = f"https://api.github.com/repos/{repo_name}/issues"
            params = {
                "state": state,
                "per_page": max_issues,
                "sort": "updated"
            }
            
            response = requests.get(issues_url, headers=headers, params=params, timeout=15)
            if response.status_code == 200:
                issues_data = response.json()
                
                for issue in issues_data:
                    # Skip pull requests
                    if issue.get("pull_request"):
                        continue
                    
                    issue_text = (issue["title"] + " " + issue.get("body", "")).lower()
                    
                    # Filter for relevant issues
                    is_relevant = any(keyword in issue_text for keyword in relevant_keywords)
                    
                    # Also check labels for relevance
                    issue_labels = [label["name"].lower() for label in issue.get("labels", [])]
                    relevant_labels = ['bug', 'performance', 'enhancement', 'feature', 'help', 'question']
                    if any(label in issue_labels for label in relevant_labels):
                        is_relevant = True
                    
                    if is_relevant:
                        issues.append({
                            "number": issue["number"],
                            "title": issue["title"],
                            "body": issue.get("body", "")[:600] if issue.get("body") else "",
                            "state": issue["state"],
                            "labels": [label["name"] for label in issue.get("labels", [])],
                            "created_at": issue["created_at"],
                            "updated_at": issue["updated_at"],
                            "url": issue["html_url"],
                            "user": issue["user"]["login"],
                            "comments": issue.get("comments", 0),
                            "relevance": "relevant"
                        })
                    
            print(f"✅ Collected {len(issues)} relevant {state} issues from {max_issues} total")
        except Exception as e:
            print(f"Error fetching filtered {state} issues: {e}")
            
        return issues

    async def _find_related_issues(self, all_issues: List[Dict], current_issue: Dict) -> List[Dict]:
        """Find issues specifically related to the current issue using AI-powered filtering"""
        related = []
        current_title = current_issue.get('issue_title', '')
        current_body = current_issue.get('issue_body', '')
        
        if not all_issues:
            return related
            
        print(f"🤖 AI-filtering {len(all_issues)} issues for relevance...")
        
        try:
            # Process issues in batches to avoid token limits
            batch_size = 5
            batches = [all_issues[i:i + batch_size] for i in range(0, len(all_issues), batch_size)]
            
            for batch in batches:
                relevance_scores = await self._get_ai_issue_relevance(current_title, current_body, batch)
                
                for i, issue in enumerate(batch):
                    if i < len(relevance_scores) and relevance_scores[i] >= 7:  # High relevance threshold
                        # Add clickable GitHub URL
                        issue_url = f"https://github.com/{issue.get('repo_name', 'microsoft/mssql-jdbc')}/issues/{issue['number']}"
                        issue["url"] = issue_url
                        issue["relevance_score"] = relevance_scores[i]
                        related.append(issue)
            
            # Sort by relevance score and return top matches
            related.sort(key=lambda x: x["relevance_score"], reverse=True)
            print(f"✅ Found {len(related)} highly relevant issues using AI filtering")
            return related[:10]  # Return top 10 most relevant issues
            
        except Exception as e:
            print(f"⚠️ AI issue filtering failed, falling back to keyword matching: {e}")
            return self._fallback_issue_filtering(all_issues, current_issue)

    async def _get_ai_issue_relevance(self, current_title: str, current_body: str, issues_batch: List[Dict]) -> List[int]:
        """Use AI to determine relevance scores for a batch of issues"""
        try:
            issues_text = ""
            for i, issue in enumerate(issues_batch):
                issues_text += f"{i+1}. Issue #{issue['number']}: {issue['title'][:100]}\n"
                if issue.get('body'):
                    issues_text += f"   Description: {issue['body'][:200]}...\n"
                issues_text += "\n"
            
            prompt = f"""Current Issue:
Title: {current_title}
Description: {current_body[:500]}

Compare this current issue with the following issues and rate their relevance on a scale of 1-10:
{issues_text}

Rate how relevant each issue is to the current issue. Consider:
- Similar technical problems or error messages
- Same components/features affected
- Related functionality or use cases
- Similar root causes or symptoms

Respond with only the numbers (1-10) separated by commas, one score per issue in order.
Example: 8,3,9,1,6"""

            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=100
            )
            
            scores_text = response.choices[0].message.content.strip()
            scores = [int(s.strip()) for s in scores_text.split(',') if s.strip().isdigit()]
            
            # Ensure we have the right number of scores
            while len(scores) < len(issues_batch):
                scores.append(1)
            
            return scores[:len(issues_batch)]
            
        except Exception as e:
            print(f"Error in AI issue relevance scoring: {e}")
            return [5] * len(issues_batch)  # Default moderate scores

    def _fallback_issue_filtering(self, all_issues: List[Dict], current_issue: Dict) -> List[Dict]:
        """Fallback keyword-based issue filtering"""
        related = []
        current_title = current_issue.get('issue_title', '').lower()
        current_body = current_issue.get('issue_body', '').lower()
        
        # General similarity keywords for better matching
        similarity_keywords = {
            'connection': ['connection', 'connect', 'network'],
            'timeout': ['timeout', 'hang', 'freeze', 'stuck', 'wait'],
            'error': ['error', 'exception', 'fail', 'crash'],
            'performance': ['slow', 'performance', 'speed', 'optimization'],
            'bug': ['bug', 'issue', 'problem', 'broken'],
            'feature': ['feature', 'enhancement', 'implement', 'add'],
            'configuration': ['config', 'setting', 'parameter', 'property'],
            'security': ['security', 'auth', 'permission', 'access']
        }
        
        # Extract current issue keywords
        current_keywords = self._extract_keywords(current_title + " " + current_body)
        
        for issue in all_issues:
            similarity_score = 0
            issue_title = issue["title"].lower()
            issue_body = issue["body"].lower()
            issue_text = issue_title + " " + issue_body
            
            # Category matching
            for category, keywords in similarity_keywords.items():
                current_has_category = any(kw in (current_title + " " + current_body) for kw in keywords)
                issue_has_category = any(kw in issue_text for kw in keywords)
                
                if current_has_category and issue_has_category:
                    similarity_score += 5  # High weight for category match
            
            # Direct keyword matching
            issue_keywords = self._extract_keywords(issue_text)
            common_keywords = set(current_keywords) & set(issue_keywords)
            similarity_score += len(common_keywords) * 2
            
            # Error pattern matching
            error_patterns = ['timeout', 'connection', 'error', 'exception', 'bug', 'fail']
            for pattern in error_patterns:
                if pattern in current_title and pattern in issue_title:
                    similarity_score += 4
                elif pattern in (current_title + current_body) and pattern in issue_text:
                    similarity_score += 2
            
            # Label-based similarity
            issue_labels = [label.lower() for label in issue["labels"]]
            important_labels = ['bug', 'performance', 'enhancement', 'feature', 'help']
            common_labels = set(important_labels) & set(issue_labels)
            similarity_score += len(common_labels) * 3
                    
            # If similarity score is high enough, it's related
            if similarity_score >= 4:  # Lower threshold for more relevant results
                # Add clickable GitHub URL
                issue_url = f"https://github.com/{issue.get('repo_name', 'microsoft/mssql-jdbc')}/issues/{issue['number']}"
                issue["url"] = issue_url
                issue["similarity_score"] = similarity_score
                related.append(issue)
        
        # Sort by similarity score and return top matches
        related.sort(key=lambda x: x["similarity_score"], reverse=True)
        return related[:10]  # Return top 10 most relevant issues

    async def _get_minimal_original_context(self, repo_name: str, headers: dict, issue_data: Dict) -> Dict:
        """Get minimal context from the original repository"""
        context = {
            "repo_languages": {},
            "repo_description": "",
            "recent_activity": []
        }
        
        try:
            # Get basic repo info
            repo_url = f"https://api.github.com/repos/{repo_name}"
            response = requests.get(repo_url, headers=headers, timeout=10)
            if response.status_code == 200:
                repo_data = response.json()
                context["repo_description"] = repo_data.get("description", "")
            
            # Get languages
            languages = await self._get_repository_languages(repo_name, headers)
            context["repo_languages"] = languages
            
            # Get just a few recent commits for minimal context
            recent_commits = await self._get_recent_commits(repo_name, headers, max_commits=5)
            context["recent_activity"] = recent_commits[:3]
            
        except Exception as e:
            print(f"Error getting minimal original context: {e}")
        
        return context

    async def _get_recent_commits(self, repo_name: str, headers: dict, max_commits: int = 20) -> List[Dict]:
        """Get recent commits with enhanced information"""
        commits = []
        
        try:
            commits_url = f"https://api.github.com/repos/{repo_name}/commits"
            params = {"per_page": max_commits}
            
            response = requests.get(commits_url, headers=headers, params=params, timeout=10)
            if response.status_code == 200:
                commit_data = response.json()
                
                for commit in commit_data:
                    commits.append({
                        "sha": commit["sha"][:7],
                        "message": commit["commit"]["message"][:200],
                        "author": commit["commit"]["author"]["name"],
                        "date": commit["commit"]["author"]["date"],
                        "url": commit["html_url"],
                        "files_changed": len(commit.get("files", []))
                    })
                    
            print(f"✅ Collected {len(commits)} recent commits")
        except Exception as e:
            print(f"Error fetching commits: {e}")
            
        return commits

    async def _get_issues_by_state(self, repo_name: str, headers: dict, state: str = "open", max_issues: int = 30) -> List[Dict]:
        """Get issues by state with enhanced filtering"""
        issues = []
        
        try:
            issues_url = f"https://api.github.com/repos/{repo_name}/issues"
            params = {
                "state": state,
                "per_page": max_issues,
                "sort": "updated"
            }
            
            response = requests.get(issues_url, headers=headers, params=params, timeout=10)
            if response.status_code == 200:
                issues_data = response.json()
                
                for issue in issues_data:
                    # Skip pull requests
                    if issue.get("pull_request"):
                        continue
                        
                    issues.append({
                        "number": issue["number"],
                        "title": issue["title"],
                        "body": issue.get("body", "")[:500] if issue.get("body") else "",
                        "state": issue["state"],
                        "labels": [label["name"] for label in issue.get("labels", [])],
                        "created_at": issue["created_at"],
                        "updated_at": issue["updated_at"],
                        "url": issue["html_url"],
                        "user": issue["user"]["login"],
                        "comments": issue.get("comments", 0),
                        "assignees": [assignee["login"] for assignee in issue.get("assignees", [])]
                    })
                    
            print(f"✅ Collected {len(issues)} {state} issues")
        except Exception as e:
            print(f"Error fetching {state} issues: {e}")
            
        return issues

    async def _get_readme_content(self, repo_name: str, headers: dict) -> Optional[str]:
        """Get README content for repository context"""
        try:
            readme_url = f"https://api.github.com/repos/{repo_name}/readme"
            response = requests.get(readme_url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                readme_data = response.json()
                readme_content = base64.b64decode(readme_data["content"]).decode('utf-8')
                return readme_content[:2000]  # Limit to first 2000 chars
        except Exception as e:
            print(f"Error fetching README: {e}")
            
        return None

    async def _get_repository_languages(self, repo_name: str, headers: dict) -> Dict[str, int]:
        """Get repository languages distribution"""
        try:
            languages_url = f"https://api.github.com/repos/{repo_name}/languages"
            response = requests.get(languages_url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"Error fetching languages: {e}")
            
        return {}

    def _find_related_issues(self, all_issues: List[Dict], current_issue: Dict) -> List[Dict]:
        """Find issues related to the current issue using enhanced matching"""
        related = []
        current_title = current_issue.get('issue_title', '').lower()
        current_body = current_issue.get('issue_body', '').lower()
        current_labels = [label.lower() for label in current_issue.get('labels', [])]
        
        # Extract keywords from current issue
        current_keywords = self._extract_keywords(current_title + " " + current_body)
        
        for issue in all_issues:
            similarity_score = 0
            issue_title = issue["title"].lower()
            issue_body = issue["body"].lower()
            issue_labels = [label.lower() for label in issue["labels"]]
            issue_text = issue_title + " " + issue_body
            
            # Keyword matching
            issue_keywords = self._extract_keywords(issue_text)
            common_keywords = set(current_keywords) & set(issue_keywords)
            similarity_score += len(common_keywords) * 2
            
            # Label matching
            common_labels = set(current_labels) & set(issue_labels)
            similarity_score += len(common_labels) * 3
            
            # Title similarity (simple word overlap)
            title_words = set(current_title.split()) & set(issue_title.split())
            similarity_score += len(title_words)
            
            # Error pattern matching
            error_patterns = ['error', 'exception', 'fail', 'bug', 'issue', 'problem']
            for pattern in error_patterns:
                if pattern in current_title and pattern in issue_title:
                    similarity_score += 2
                    
            # If similarity score is high enough, it's related
            if similarity_score >= 3:
                issue["similarity_score"] = similarity_score
                related.append(issue)
        
        # Sort by similarity score and return top matches
        related.sort(key=lambda x: x["similarity_score"], reverse=True)
        return related[:8]  # Return top 8 related issues

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text for similarity matching"""
        import re
        
        # Simple keyword extraction
        words = re.findall(r'\b\w+\b', text.lower())
        # Filter out common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'can', 'may', 'might', 'must', 'shall', 'this', 'that', 'these', 'those'}
        keywords = [word for word in words if len(word) > 3 and word not in stop_words]
        return keywords[:10]  # Return top 10 keywords

    def _detect_issue_type(self, issue_data: Dict[str, Any]) -> str:
        """Detect issue type from labels and content for parameter tuning"""
        
        labels = [label.lower() for label in issue_data.get('labels', [])]
        title = issue_data.get('issue_title', '').lower()
        body = issue_data.get('issue_body', '').lower()
        
        # Check labels first (highest priority)
        if any(label in ['bug', 'error', 'broken', 'fix', 'defect'] for label in labels):
            return "bug"
        elif any(label in ['feature', 'enhancement', 'improvement', 'feature-request'] for label in labels):
            return "feature"
        elif any(label in ['documentation', 'docs', 'readme', 'wiki'] for label in labels):
            return "documentation"
        elif any(label in ['question', 'help', 'support', 'help-wanted'] for label in labels):
            return "question"
        
        # Check title and body keywords
        bug_keywords = ['error', 'bug', 'broken', 'crash', 'fail', 'issue', 'problem', 'exception', 'not working']
        feature_keywords = ['feature', 'add', 'implement', 'enhance', 'improvement', 'support for', 'would like']
        doc_keywords = ['documentation', 'docs', 'readme', 'guide', 'tutorial', 'example', 'how to']
        question_keywords = ['how', 'why', 'what', 'question', 'help', 'usage', 'clarification']
        
        text = title + " " + body
        
        if any(keyword in text for keyword in bug_keywords):
            return "bug"
        elif any(keyword in text for keyword in feature_keywords):
            return "feature"
        elif any(keyword in text for keyword in doc_keywords):
            return "documentation"
        elif any(keyword in text for keyword in question_keywords):
            return "question"
        
        return "default"

    def _get_system_prompt(self, issue_type: str) -> str:
        """Get specialized system prompt based on issue type"""
        
        base_prompt = "You are Git.AI, an expert software development assistant and GitHub issue resolution specialist. You have deep expertise in software engineering, debugging, problem-solving, and providing comprehensive technical solutions. "
        
        type_specific = {
            "bug": "You excel at diagnosing software issues, analyzing error patterns, debugging problems, and providing systematic troubleshooting solutions. Focus on root cause analysis with specific technical solutions and industry best practices.",
            "feature": "You specialize in feature design and software development planning. Focus on implementation strategies, technical architecture, and development best practices.",
            "documentation": "You specialize in technical documentation and software guides. Focus on clear examples, configuration details, setup instructions, and comprehensive troubleshooting guides.",
            "question": "You provide expert guidance on software development, technical implementation, and best practices. Focus on practical solutions, examples, and proven implementation patterns.",
            "default": "You provide comprehensive technical analysis and software development solutions with detailed implementation guidance and industry best practices."
        }
        
        expertise_note = " Your responses leverage extensive knowledge of software repositories, known patterns, proven solutions, and the latest development practices. Always provide specific, actionable solutions with relevant code examples and configuration details."
        
        return base_prompt + type_specific.get(issue_type, type_specific["default"]) + expertise_note

    def _build_enhanced_prompt(self, issue_data: Dict[str, Any], repo_context: Dict[str, Any], issue_type: str) -> str:
        """Build comprehensive prompt using relevant repository context"""
        
        prompt_parts = []
        
        prompt_parts.append("**COMPREHENSIVE TECHNICAL ANALYSIS**\n")
        
        original_repo = issue_data.get('repo_name', 'Unknown')
        is_reference_repo = repo_context.get('is_reference_repo', False)
        
        prompt_parts.append(f"**Original Issue Repository:** {original_repo}")
        prompt_parts.append(f"**Issue Type:** {issue_type.upper()}")
        prompt_parts.append(f"**Reference Repository Context:** {'DIRECT' if is_reference_repo else 'CROSS-REFERENCE'}")
        prompt_parts.append("")
        
        # Reference Repository Intelligence
        reference_commits = repo_context.get('reference_commits', [])
        reference_open_issues = repo_context.get('reference_open_issues', [])
        reference_closed_issues = repo_context.get('reference_closed_issues', [])
        related_issues = repo_context.get('related_issues', [])
        
        prompt_parts.append("**Reference Repository Intelligence:**")
        prompt_parts.append(f"- Relevant Commits Analyzed: {len(reference_commits)}")
        prompt_parts.append(f"- Related Open Issues: {len(reference_open_issues)}")
        prompt_parts.append(f"- Related Resolved Issues: {len(reference_closed_issues)}")
        prompt_parts.append(f"- Highly Relevant Issues: {len(related_issues)}")
        prompt_parts.append("")
        
        # Current issue analysis
        prompt_parts.append("**Current Issue Analysis:**")
        prompt_parts.append(f"- Issue #{issue_data.get('issue_number', 'N/A')}: {issue_data.get('issue_title', 'No title')}")
        prompt_parts.append(f"- Reporter: {issue_data.get('creator', 'Unknown')} ({issue_data.get('creator_type', 'Unknown')})")
        if issue_data.get('labels'):
            prompt_parts.append(f"- Labels: {', '.join(issue_data['labels'])}")
        prompt_parts.append(f"- Description: {issue_data.get('issue_body', 'No description provided')}")
        prompt_parts.append("")

        # Original repository context (if different from reference repo)
        if not is_reference_repo:
            original_context = repo_context.get('original_repo_context', {})
            if original_context:
                prompt_parts.append(f"**Original Repository Context ({original_repo}):**")
                if original_context.get('repo_description'):
                    prompt_parts.append(f"- Description: {original_context['repo_description']}")
                if original_context.get('repo_languages'):
                    languages = list(original_context['repo_languages'].keys())
                    prompt_parts.append(f"- Primary Languages: {', '.join(languages[:3])}")
                prompt_parts.append("")

        # Highly relevant issues
        if related_issues:
            prompt_parts.append("**Most Relevant Reference Repository Issues (AI-filtered):**")
            for issue in related_issues[:6]:
                state_indicator = "🔴 OPEN" if issue['state'] == 'open' else "✅ RESOLVED"
                default_url = f"https://github.com/microsoft/mssql-jdbc/issues/{issue['number']}"
                issue_url = issue.get('url', default_url)
                issue_link = f"[#{issue['number']}]({issue_url})"
                relevance_score = issue.get('relevance_score', issue.get('similarity_score', 'N/A'))
                prompt_parts.append(f"- {state_indicator} {issue_link}: {issue['title']} (AI relevance: {relevance_score}/10)")
                if issue.get('body') and len(issue['body']) > 50:
                    prompt_parts.append(f"  Context: {issue['body'][:180]}...")
                if issue['state'] == 'closed':
                    prompt_parts.append(f"  🎯 PROVEN SOLUTION AVAILABLE - analyze resolution patterns")
                prompt_parts.append("")

        # Recent development activity
        if reference_commits:
            prompt_parts.append("**Recent Reference Repository Development Activity:**")
            ai_filtered_count = len([c for c in reference_commits if c.get('ai_filtered', False)])
            prompt_parts.append(f"📋 Recent relevant commits (AI-filtered: {ai_filtered_count}/{len(reference_commits)}):")
            for commit in reference_commits[:6]:
                commit_link = f"[{commit['sha']}]({commit['url']})"
                ai_badge = " 🤖" if commit.get('ai_filtered', False) else ""
                prompt_parts.append(f"- {commit_link}: {commit['message'][:120]}... (by {commit['author']}){ai_badge}")
            prompt_parts.append("")

        # Current issues landscape
        if reference_open_issues:
            prompt_parts.append("**Current Reference Repository Issues Status:**")
            # Focus on high-impact open issues
            high_impact_issues = [issue for issue in reference_open_issues[:15] 
                                if issue.get('comments', 0) > 2 or 
                                   any(label in ['bug', 'performance', 'critical'] 
                                       for label in [l.lower() for l in issue.get('labels', [])])]
            
            if high_impact_issues:
                prompt_parts.append("⚠️ High-impact open issues (community attention):")
                for issue in high_impact_issues[:4]:
                    prompt_parts.append(f"- #{issue['number']}: {issue['title']} ({issue['comments']} comments)")
            prompt_parts.append("")

        # Resolution patterns from closed issues
        if reference_closed_issues:
            prompt_parts.append("**Reference Repository Resolution Patterns:**")
            # Focus on recently resolved issues with good solutions
            recent_resolved = [issue for issue in reference_closed_issues[:20] 
                             if issue.get('comments', 0) > 1]
            
            if recent_resolved:
                prompt_parts.append("✅ Recently resolved issues (proven solutions):")
                for issue in recent_resolved[:4]:
                    prompt_parts.append(f"- #{issue['number']}: {issue['title']} ({issue['comments']} comments)")
                    if issue.get('body'):
                        prompt_parts.append(f"  Solution context: {issue['body'][:150]}...")
            prompt_parts.append("")

        # Documentation context
        if repo_context.get('readme_content'):
            prompt_parts.append("**Reference Repository Official Documentation:**")
            prompt_parts.append(f"```\n{repo_context['readme_content'][:1200]}...\n```")
            prompt_parts.append("")

        # Expert resolution task
        prompt_parts.append("**EXPERT TECHNICAL RESOLUTION TASK:**")
        prompt_parts.append("Using your software development expertise and the comprehensive analysis above, provide a DETAILED, EXPERT-LEVEL solution:")
        prompt_parts.append("")
        prompt_parts.append("1. **Root Cause Analysis** (2-3 paragraphs)")
        prompt_parts.append("   - Apply technical expertise to identify the core issue")
        prompt_parts.append("   - Reference specific patterns from related issues above")
        prompt_parts.append("   - Connect to known software development patterns")
        prompt_parts.append("")
        prompt_parts.append("2. **Technical Solution Steps** (6-10 detailed steps)")
        prompt_parts.append("   - Specific implementation guidance")
        prompt_parts.append("   - Include exact configuration parameters and values")
        prompt_parts.append("   - Reference lessons from resolved issues above")
        prompt_parts.append("")
        prompt_parts.append("3. **Code Examples & Configuration** (Multiple examples)")
        prompt_parts.append("   - Relevant code snippets and examples")
        prompt_parts.append("   - Configuration examples")
        prompt_parts.append("   - Implementation patterns and best practices")
        prompt_parts.append("   - Error handling and validation examples")
        prompt_parts.append("")
        prompt_parts.append("4. **Technical Context** (1-2 paragraphs)")
        prompt_parts.append("   - How recent developments inform this solution")
        prompt_parts.append("   - Specific technical considerations")
        prompt_parts.append("   - Version compatibility and recommendations")
        prompt_parts.append("")
        
        prompt_parts.append("**EXPERT OUTPUT REQUIREMENTS:**")
        prompt_parts.append("- COMPREHENSIVE EXPERT RESPONSE (aim for 4500+ characters)")
        prompt_parts.append("- Reference ONLY highly relevant issue numbers and commit links from analysis above")
        prompt_parts.append("- Use markdown format: [commit_sha](commit_url) for clickable commit references")
        prompt_parts.append("- Verify commit relevance before referencing - only include commits directly related to the issue")
        prompt_parts.append("- Include multiple practical code examples")
        prompt_parts.append("- Provide expert-level implementation guidance with exact parameters")
        prompt_parts.append("- Connect solution to proven patterns and best practices")
        prompt_parts.append("- Each section should demonstrate deep technical expertise and be immediately actionable")
        prompt_parts.append("- Focus on proven implementations and solutions from the reference repository")
        
        return "\n".join(prompt_parts)

    def _generate_enhanced_fallback_response(self, issue_data: Dict[str, Any], repo_context: Dict[str, Any]) -> str:
        """Generate enhanced software development-focused fallback response with repository context"""
        
        related_issues_count = len(repo_context.get('related_issues', []))
        reference_commits_count = len(repo_context.get('reference_commits', []))
        reference_open_count = len(repo_context.get('reference_open_issues', []))
        reference_closed_count = len(repo_context.get('reference_closed_issues', []))
        is_reference_repo = repo_context.get('is_reference_repo', False)
        
        # Get some specific issue details for better context
        issue_title = issue_data.get('issue_title', 'Unknown Issue')
        issue_body = issue_data.get('issue_body', '')
        repo_name = issue_data.get('repo_name', 'Unknown')
        
        # Extract relevant issues for reference
        related_issues_text = ""
        if repo_context.get('related_issues'):
            related_issues_text = "\n**🔗 Related Repository Issues:**\n"
            for issue in repo_context['related_issues'][:3]:
                state = "✅ RESOLVED" if issue['state'] == 'closed' else "🔴 OPEN"
                related_issues_text += f"- {state} #{issue['number']}: {issue['title'][:80]}...\n"
                if issue['state'] == 'closed':
                    related_issues_text += f"  💡 *This was successfully resolved - check solution patterns*\n"
        
        return f"""**🔧 Expert Software Development Analysis**

**Repository Context:** {repo_name}
**Issue:** #{issue_data.get('issue_number', 'N/A')} - {issue_title}
**Development Intelligence:** {'DIRECT ANALYSIS' if is_reference_repo else 'CROSS-REFERENCE EXPERTISE'}

**� Microsoft MSSQL JDBC Repository Intelligence:**
- 🔍 Analyzed {mssql_commits_count} JDBC-relevant commits
- 📋 Found {related_jdbc_count} highly relevant JDBC issues  
- 🟢 {mssql_closed_count} resolved JDBC issues (proven solutions available)
- 🔴 {mssql_open_count} current JDBC issues (active development)
{related_issues_text}
**🎯 MSSQL JDBC Driver Expert Resolution Steps:**

**1. JDBC Driver Root Cause Analysis**
   - Analyze connection string patterns and driver configuration
   - Check Microsoft SQL Server compatibility matrix for driver versions
   - Review JDBC URL parameters and authentication mechanisms
   - Examine connection pool configuration (HikariCP, DBCP2, etc.)

**2. Microsoft JDBC-Specific Investigation**
   - Test with latest Microsoft MSSQL JDBC driver version (12.8.1+)
   - Verify SQL Server instance configuration and network connectivity
   - Check for TLS/SSL certificate issues and encryption settings
   - Review SQL Server authentication modes (Windows Auth vs SQL Auth)

**3. JDBC Configuration Optimization**
   ```java
   // Recommended Microsoft JDBC connection string
   String url = "jdbc:sqlserver://server:port;databaseName=db;" +
                "encrypt=true;trustServerCertificate=false;" +
                "hostNameInCertificate=*.database.windows.net;" +
                "loginTimeout=30;";
   ```

**4. Connection Pool & Driver Setup**
   ```java
   // HikariCP with Microsoft JDBC Driver
   HikariConfig config = new HikariConfig();
   config.setJdbcUrl(connectionUrl);
   config.setDriverClassName("com.microsoft.sqlserver.jdbc.SQLServerDriver");
   config.setMaximumPoolSize(20);
   config.setConnectionTimeout(30000);
   config.setIdleTimeout(600000);
   config.setMaxLifetime(1800000);
   ```

**5. Microsoft SQL Server Connectivity Validation**
   - Test direct connectivity using SQL Server Management Studio
   - Verify firewall rules and network security groups
   - Check SQL Server TCP/IP protocol enablement
   - Validate connection from application server to SQL Server

**6. JDBC Driver Error Pattern Analysis**
   - Review SQL Server error logs for connection attempts
   - Check application logs for JDBC driver exceptions
   - Monitor connection pool metrics and timeout patterns
   - Analyze network latency and packet loss

**7. Microsoft JDBC Best Practices Implementation**
   - Enable connection validation with isValid() method
   - Configure appropriate statement and connection timeouts
   - Implement retry logic for transient connection failures
   - Use prepared statements for SQL injection prevention

**8. Performance & Monitoring Setup**
   - Configure JDBC driver logging for troubleshooting
   - Implement connection pool monitoring (JMX, metrics)
   - Set up SQL Server performance counter monitoring
   - Enable query execution plan analysis

**🔍 Microsoft JDBC Context from Repository Analysis:**
Based on analysis of {mssql_commits_count} recent JDBC commits and {related_jdbc_count} related issues, common patterns include:
- Connection string configuration issues (authentication, encryption)
- Driver version compatibility with SQL Server versions
- Connection pool exhaustion and timeout problems
- TLS/SSL certificate validation challenges

**✅ Validation & Testing Strategy:**
1. **JDBC Connection Test**: Create minimal test application with Microsoft JDBC driver
2. **Load Testing**: Simulate concurrent connections matching production load
3. **Failover Testing**: Test connection resilience during SQL Server maintenance
4. **Performance Baseline**: Establish connection time and query execution metrics

**📚 Microsoft JDBC Documentation References:**
- Official Microsoft JDBC Driver documentation
- SQL Server connection best practices
- Azure SQL Database connectivity guidelines
- JDBC driver version compatibility matrix

*This analysis leverages expertise from {related_issues_count} related issues and {reference_commits_count} recent development activities.*

---
**Generated by Git.AI Enhanced Assistant with Software Development Specialization**
"""

    # Keep original prompt building method for compatibility
    def _build_resolution_prompt(self, issue_data: Dict[str, Any]) -> str:
        """Build a structured prompt for AI resolution generation (original method)"""
        
        return f"""
**GITHUB ISSUE ANALYSIS REQUEST**

Repository: {issue_data['repo_name']}
Language: {issue_data.get('language', 'Unknown')}
Issue #{issue_data['issue_number']}: {issue_data['issue_title']}

**Issue Description:**
{issue_data['issue_body'] or 'No description provided'}

**Context:**
- Repository Description: {issue_data.get('repo_description', 'No description')}
- Labels: {', '.join(issue_data['labels']) if issue_data['labels'] else 'None'}
- Topics: {', '.join(issue_data['repo_topics']) if issue_data['repo_topics'] else 'None'}
- Reporter: {issue_data['creator']} ({issue_data['creator_type']})

**TASK:**
Provide only a numbered list of 3-5 specific, actionable resolution steps for this GitHub issue. Focus on practical steps a maintainer can take to resolve this specific problem. Do not include classifications, assessments, or response templates - just the resolution steps.

Format as:
1. Step one
2. Step two
3. Step three
etc.
"""

    # Keep original fallback method for compatibility
    def _generate_fallback_response(self, issue_data: Dict[str, Any]) -> str:
        """Generate a simple template-based response as fallback"""
        return f"""
**Resolution Steps for Issue #{issue_data['issue_number']}**

1. **Reproduce the issue**: Try to replicate the problem described
2. **Check documentation**: Review README and relevant docs for similar cases
3. **Search similar issues**: Look for related issues in the repository history
4. **Analyze code context**: Check the files/components mentioned in the issue
5. **Provide solution**: Based on findings, suggest a fix or workaround

*Generated by Git.AI Assistant*
"""
    
    def _build_resolution_prompt(self, issue_data: Dict[str, Any]) -> str:
        """Build a structured prompt for AI resolution generation"""
        
        return f"""
**GITHUB ISSUE ANALYSIS REQUEST**

Repository: {issue_data['repo_name']}
Language: {issue_data.get('language', 'Unknown')}
Issue #{issue_data['issue_number']}: {issue_data['issue_title']}

**Issue Description:**
{issue_data['issue_body'] or 'No description provided'}

**Context:**
- Repository Description: {issue_data.get('repo_description', 'No description')}
- Labels: {', '.join(issue_data['labels']) if issue_data['labels'] else 'None'}
- Topics: {', '.join(issue_data['repo_topics']) if issue_data['repo_topics'] else 'None'}
- Reporter: {issue_data['creator']} ({issue_data['creator_type']})

**TASK:**
Provide only a numbered list of 3-5 specific, actionable resolution steps for this GitHub issue. Focus on practical steps a maintainer can take to resolve this specific problem. Do not include classifications, assessments, or response templates - just the resolution steps.

Format as:
1. Step one
2. Step two
3. Step three
etc.
"""
