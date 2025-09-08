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
        """MSSQL JDBC-focused repository context collection with relevant data only"""
        
        # Always target the Microsoft MSSQL JDBC repository for enhanced context
        target_repo = "microsoft/mssql-jdbc"
        original_repo = issue_data.get('repo_name', '')
        
        context = {
            "mssql_jdbc_commits": [],
            "mssql_jdbc_open_issues": [],
            "mssql_jdbc_closed_issues": [],
            "related_jdbc_issues": [],
            "jdbc_readme_content": None,
            "original_repo_context": {},
            "is_mssql_jdbc_repo": original_repo == target_repo
        }
        
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "Git-AI-Bot"
        }
        
        # Use the GitHub token for better rate limits
        if self.github_token:
            headers["Authorization"] = f"token {self.github_token}"
        
        try:
            print(f"🔍 Collecting MSSQL JDBC repository context from: {target_repo}")
            print(f"📋 Original issue repository: {original_repo}")
            
            # Get MSSQL JDBC recent commits (filter for relevant ones)
            print("📥 Fetching MSSQL JDBC recent commits...")
            mssql_commits = await self._get_filtered_commits(target_repo, headers, issue_data, max_commits=30)
            context["mssql_jdbc_commits"] = mssql_commits

            # Get MSSQL JDBC open issues (filter for relevant ones)
            print("📥 Fetching MSSQL JDBC open issues...")
            mssql_open_issues = await self._get_filtered_issues(target_repo, headers, "open", issue_data, max_issues=50)
            context["mssql_jdbc_open_issues"] = mssql_open_issues

            # Get MSSQL JDBC closed issues (filter for relevant ones)
            print("📥 Fetching MSSQL JDBC closed issues...")
            mssql_closed_issues = await self._get_filtered_issues(target_repo, headers, "closed", issue_data, max_issues=100)
            context["mssql_jdbc_closed_issues"] = mssql_closed_issues

            # Find JDBC-specific related issues
            print("🔍 Finding JDBC-related issues...")
            all_jdbc_issues = mssql_open_issues + mssql_closed_issues
            related_jdbc_issues = self._find_jdbc_related_issues(all_jdbc_issues, issue_data)
            context["related_jdbc_issues"] = related_jdbc_issues
            
            # Get MSSQL JDBC README for context
            print("📄 Fetching MSSQL JDBC README...")
            jdbc_readme = await self._get_readme_content(target_repo, headers)
            context["jdbc_readme_content"] = jdbc_readme
            
            # If the original issue is NOT from mssql-jdbc repo, get minimal context from original repo
            if not context["is_mssql_jdbc_repo"]:
                print(f"📄 Getting minimal context from original repo: {original_repo}")
                original_context = await self._get_minimal_original_context(original_repo, headers, issue_data)
                context["original_repo_context"] = original_context
                
        except Exception as e:
            print(f"Error collecting MSSQL JDBC repository context: {str(e)}")
        
        return context

    async def _get_filtered_commits(self, repo_name: str, headers: dict, issue_data: Dict, max_commits: int = 30) -> List[Dict]:
        """Get commits filtered for JDBC/connection relevance"""
        commits = []
        
        # JDBC-specific keywords to filter commits
        jdbc_keywords = ['jdbc', 'connection', 'pool', 'timeout', 'sql', 'driver', 'database', 'hikari', 'dbcp', 'leak', 'close', 'statement', 'resultset', 'transaction']
        
        try:
            commits_url = f"https://api.github.com/repos/{repo_name}/commits"
            params = {"per_page": max_commits}
            
            response = requests.get(commits_url, headers=headers, params=params, timeout=15)
            if response.status_code == 200:
                commit_data = response.json()
                
                for commit in commit_data:
                    commit_message = commit["commit"]["message"].lower()
                    
                    # Filter for JDBC-relevant commits
                    is_relevant = any(keyword in commit_message for keyword in jdbc_keywords)
                    
                    # Also include commits that match current issue keywords
                    issue_keywords = self._extract_keywords(issue_data.get('issue_title', '') + ' ' + issue_data.get('issue_body', ''))
                    if any(keyword in commit_message for keyword in issue_keywords[:5]):
                        is_relevant = True
                    
                    if is_relevant:
                        commits.append({
                            "sha": commit["sha"][:7],
                            "message": commit["commit"]["message"][:250],
                            "author": commit["commit"]["author"]["name"],
                            "date": commit["commit"]["author"]["date"],
                            "url": commit["html_url"],
                            "relevance": "jdbc-specific"
                        })
                    
            print(f"✅ Collected {len(commits)} JDBC-relevant commits from {max_commits} total")
        except Exception as e:
            print(f"Error fetching filtered commits: {e}")
            
        return commits

    async def _get_filtered_issues(self, repo_name: str, headers: dict, state: str, issue_data: Dict, max_issues: int = 50) -> List[Dict]:
        """Get issues filtered for JDBC/connection relevance"""
        issues = []
        
        # JDBC-specific keywords and patterns
        jdbc_keywords = ['jdbc', 'connection', 'pool', 'timeout', 'sql', 'driver', 'database', 'hikari', 'dbcp', 'leak', 'statement', 'resultset', 'transaction', 'deadlock', 'performance']
        
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
                    
                    # Filter for JDBC-relevant issues
                    is_relevant = any(keyword in issue_text for keyword in jdbc_keywords)
                    
                    # Also check labels for relevance
                    issue_labels = [label["name"].lower() for label in issue.get("labels", [])]
                    relevant_labels = ['bug', 'performance', 'connection', 'timeout', 'leak', 'pool']
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
                            "relevance": "jdbc-specific"
                        })
                    
            print(f"✅ Collected {len(issues)} JDBC-relevant {state} issues from {max_issues} total")
        except Exception as e:
            print(f"Error fetching filtered {state} issues: {e}")
            
        return issues

    def _find_jdbc_related_issues(self, all_issues: List[Dict], current_issue: Dict) -> List[Dict]:
        """Find JDBC issues specifically related to the current issue"""
        related = []
        current_title = current_issue.get('issue_title', '').lower()
        current_body = current_issue.get('issue_body', '').lower()
        
        # JDBC-specific similarity keywords
        jdbc_similarity_keywords = {
            'connection': ['connection', 'connect', 'pool', 'datasource'],
            'timeout': ['timeout', 'hang', 'freeze', 'stuck', 'wait'],
            'leak': ['leak', 'memory', 'close', 'resource'],
            'performance': ['slow', 'performance', 'speed', 'optimization'],
            'error': ['error', 'exception', 'fail', 'crash'],
            'transaction': ['transaction', 'commit', 'rollback', 'isolation'],
            'security': ['security', 'ssl', 'tls', 'encrypt', 'auth'],
            'configuration': ['config', 'setting', 'parameter', 'property']
        }
        
        # Extract current issue keywords
        current_keywords = self._extract_keywords(current_title + " " + current_body)
        
        for issue in all_issues:
            similarity_score = 0
            issue_title = issue["title"].lower()
            issue_body = issue["body"].lower()
            issue_text = issue_title + " " + issue_body
            
            # JDBC category matching
            for category, keywords in jdbc_similarity_keywords.items():
                current_has_category = any(kw in (current_title + " " + current_body) for kw in keywords)
                issue_has_category = any(kw in issue_text for kw in keywords)
                
                if current_has_category and issue_has_category:
                    similarity_score += 5  # High weight for category match
            
            # Direct keyword matching
            issue_keywords = self._extract_keywords(issue_text)
            common_keywords = set(current_keywords) & set(issue_keywords)
            similarity_score += len(common_keywords) * 2
            
            # Error pattern matching (specific to JDBC)
            jdbc_error_patterns = ['timeout', 'connection', 'pool', 'leak', 'deadlock', 'transaction', 'statement']
            for pattern in jdbc_error_patterns:
                if pattern in current_title and pattern in issue_title:
                    similarity_score += 4
                elif pattern in (current_title + current_body) and pattern in issue_text:
                    similarity_score += 2
            
            # Label-based similarity
            issue_labels = [label.lower() for label in issue["labels"]]
            important_labels = ['bug', 'performance', 'connection', 'timeout', 'enhancement']
            common_labels = set(important_labels) & set(issue_labels)
            similarity_score += len(common_labels) * 3
                    
            # If similarity score is high enough, it's related
            if similarity_score >= 4:  # Lower threshold for more relevant results
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
        """Get MSSQL JDBC specialized system prompt based on issue type"""
        
        base_prompt = "You are Git.AI, an expert Microsoft SQL Server JDBC driver specialist and GitHub issue resolution assistant. You have deep expertise in JDBC connectivity, connection pooling, SQL Server integration, and Microsoft's mssql-jdbc driver implementation. "
        
        type_specific = {
            "bug": "You excel at diagnosing JDBC connection issues, SQL Server connectivity problems, connection pool leaks, timeout issues, and driver-specific bugs. Focus on systematic troubleshooting with specific JDBC driver solutions and Microsoft SQL Server best practices.",
            "feature": "You specialize in JDBC feature design and SQL Server integration planning. Focus on Microsoft JDBC driver capabilities, SQL Server feature compatibility, and performance optimization strategies.",
            "documentation": "You specialize in JDBC documentation and SQL Server connectivity guides. Focus on clear examples, connection string configurations, driver setup, and troubleshooting guides specific to Microsoft environments.",
            "question": "You provide expert guidance on Microsoft JDBC driver usage, SQL Server connectivity, and best practices. Focus on practical solutions, configuration examples, and proven implementation patterns.",
            "default": "You provide comprehensive JDBC driver analysis and Microsoft SQL Server connectivity solutions with detailed implementation guidance and industry best practices."
        }
        
        jdbc_expertise = " Your responses leverage extensive knowledge of the Microsoft mssql-jdbc repository, known issues, proven solutions, and the latest driver developments. Always provide specific, actionable solutions with relevant code examples and configuration details."
        
        return base_prompt + type_specific.get(issue_type, type_specific["default"]) + jdbc_expertise

    def _build_enhanced_prompt(self, issue_data: Dict[str, Any], repo_context: Dict[str, Any], issue_type: str) -> str:
        """Build MSSQL JDBC-focused prompt using relevant repository context"""
        
        prompt_parts = []
        
        prompt_parts.append("**MICROSOFT SQL SERVER JDBC DRIVER EXPERT ANALYSIS**\n")
        
        original_repo = issue_data.get('repo_name', 'Unknown')
        is_mssql_jdbc = repo_context.get('is_mssql_jdbc_repo', False)
        
        prompt_parts.append(f"**Original Issue Repository:** {original_repo}")
        prompt_parts.append(f"**Issue Type:** {issue_type.upper()}")
        prompt_parts.append(f"**MSSQL JDBC Expert Context:** {'DIRECT' if is_mssql_jdbc else 'CROSS-REFERENCE'}")
        prompt_parts.append("")
        
        # MSSQL JDBC Repository Intelligence
        mssql_commits = repo_context.get('mssql_jdbc_commits', [])
        mssql_open_issues = repo_context.get('mssql_jdbc_open_issues', [])
        mssql_closed_issues = repo_context.get('mssql_jdbc_closed_issues', [])
        related_jdbc_issues = repo_context.get('related_jdbc_issues', [])
        
        prompt_parts.append("**Microsoft MSSQL JDBC Repository Intelligence:**")
        prompt_parts.append(f"- JDBC-Relevant Commits Analyzed: {len(mssql_commits)}")
        prompt_parts.append(f"- JDBC-Related Open Issues: {len(mssql_open_issues)}")
        prompt_parts.append(f"- JDBC-Related Resolved Issues: {len(mssql_closed_issues)}")
        prompt_parts.append(f"- Highly Relevant JDBC Issues: {len(related_jdbc_issues)}")
        prompt_parts.append("")
        
        # Current issue analysis
        prompt_parts.append("**Current Issue Analysis:**")
        prompt_parts.append(f"- Issue #{issue_data.get('issue_number', 'N/A')}: {issue_data.get('issue_title', 'No title')}")
        prompt_parts.append(f"- Reporter: {issue_data.get('creator', 'Unknown')} ({issue_data.get('creator_type', 'Unknown')})")
        if issue_data.get('labels'):
            prompt_parts.append(f"- Labels: {', '.join(issue_data['labels'])}")
        prompt_parts.append(f"- Description: {issue_data.get('issue_body', 'No description provided')}")
        prompt_parts.append("")

        # Original repository context (if different from mssql-jdbc)
        if not is_mssql_jdbc:
            original_context = repo_context.get('original_repo_context', {})
            if original_context:
                prompt_parts.append(f"**Original Repository Context ({original_repo}):**")
                if original_context.get('repo_description'):
                    prompt_parts.append(f"- Description: {original_context['repo_description']}")
                if original_context.get('repo_languages'):
                    languages = list(original_context['repo_languages'].keys())
                    prompt_parts.append(f"- Primary Languages: {', '.join(languages[:3])}")
                prompt_parts.append("")

        # Highly relevant JDBC issues
        if related_jdbc_issues:
            prompt_parts.append("**Most Relevant Microsoft MSSQL JDBC Issues:**")
            for issue in related_jdbc_issues[:6]:
                state_indicator = "🔴 OPEN" if issue['state'] == 'open' else "✅ RESOLVED"
                prompt_parts.append(f"- {state_indicator} #{issue['number']}: {issue['title']} (relevance: {issue['similarity_score']})")
                if issue.get('body') and len(issue['body']) > 50:
                    prompt_parts.append(f"  Context: {issue['body'][:180]}...")
                if issue['state'] == 'closed':
                    prompt_parts.append(f"  🎯 PROVEN SOLUTION AVAILABLE - analyze resolution patterns")
                prompt_parts.append("")

        # Recent JDBC development activity
        if mssql_commits:
            prompt_parts.append("**Recent Microsoft MSSQL JDBC Development Activity:**")
            prompt_parts.append("📋 Recent JDBC-relevant commits:")
            for commit in mssql_commits[:6]:
                prompt_parts.append(f"- {commit['sha']}: {commit['message'][:120]}... (by {commit['author']})")
            prompt_parts.append("")

        # Current JDBC issues landscape
        if mssql_open_issues:
            prompt_parts.append("**Current Microsoft MSSQL JDBC Issues Status:**")
            # Focus on high-impact open issues
            high_impact_issues = [issue for issue in mssql_open_issues[:15] 
                                if issue.get('comments', 0) > 2 or 
                                   any(label in ['bug', 'performance', 'critical'] 
                                       for label in [l.lower() for l in issue.get('labels', [])])]
            
            if high_impact_issues:
                prompt_parts.append("⚠️ High-impact open JDBC issues (community attention):")
                for issue in high_impact_issues[:4]:
                    prompt_parts.append(f"- #{issue['number']}: {issue['title']} ({issue['comments']} comments)")
            prompt_parts.append("")

        # MSSQL JDBC resolved patterns
        if mssql_closed_issues:
            prompt_parts.append("**Microsoft MSSQL JDBC Resolution Patterns:**")
            # Focus on recently resolved issues with good solutions
            recent_resolved = [issue for issue in mssql_closed_issues[:20] 
                             if issue.get('comments', 0) > 1]
            
            if recent_resolved:
                prompt_parts.append("✅ Recently resolved JDBC issues (proven solutions):")
                for issue in recent_resolved[:4]:
                    prompt_parts.append(f"- #{issue['number']}: {issue['title']} ({issue['comments']} comments)")
                    if issue.get('body'):
                        prompt_parts.append(f"  Solution context: {issue['body'][:150]}...")
            prompt_parts.append("")

        # MSSQL JDBC Documentation context
        if repo_context.get('jdbc_readme_content'):
            prompt_parts.append("**Microsoft MSSQL JDBC Official Documentation:**")
            prompt_parts.append(f"```\n{repo_context['jdbc_readme_content'][:1200]}...\n```")
            prompt_parts.append("")

        # Expert resolution task
        prompt_parts.append("**EXPERT MSSQL JDBC RESOLUTION TASK:**")
        prompt_parts.append("Using your Microsoft SQL Server JDBC driver expertise and the comprehensive analysis above, provide a DETAILED, EXPERT-LEVEL solution:")
        prompt_parts.append("")
        prompt_parts.append("1. **JDBC Driver Root Cause Analysis** (2-3 paragraphs)")
        prompt_parts.append("   - Apply MSSQL JDBC expertise to identify the core issue")
        prompt_parts.append("   - Reference specific JDBC driver patterns from related issues above")
        prompt_parts.append("   - Connect to known Microsoft SQL Server connectivity patterns")
        prompt_parts.append("")
        prompt_parts.append("2. **Microsoft JDBC Driver Solution Steps** (6-10 detailed steps)")
        prompt_parts.append("   - Specific to Microsoft mssql-jdbc driver implementation")
        prompt_parts.append("   - Include exact configuration parameters and values")
        prompt_parts.append("   - Reference lessons from resolved JDBC issues above")
        prompt_parts.append("")
        prompt_parts.append("3. **JDBC Code Examples & Configuration** (Multiple examples)")
        prompt_parts.append("   - Connection string examples specific to SQL Server")
        prompt_parts.append("   - Driver configuration examples")
        prompt_parts.append("   - Connection pool configuration (HikariCP, DBCP, etc.)")
        prompt_parts.append("   - Error handling and retry logic examples")
        prompt_parts.append("")
        prompt_parts.append("4. **Microsoft JDBC Driver Context** (1-2 paragraphs)")
        prompt_parts.append("   - How recent JDBC driver developments inform this solution")
        prompt_parts.append("   - Specific Microsoft SQL Server considerations")
        prompt_parts.append("   - Driver version compatibility and recommendations")
        prompt_parts.append("")
        prompt_parts.append("5. **JDBC Best Practices & Prevention** (5-7 recommendations)")
        prompt_parts.append("   - Microsoft JDBC driver-specific best practices")
        prompt_parts.append("   - SQL Server connectivity optimization")
        prompt_parts.append("   - Connection pool tuning and monitoring")
        prompt_parts.append("   - Common pitfall prevention")
        prompt_parts.append("")
        prompt_parts.append("6. **JDBC Testing & Validation** (4-6 approaches)")
        prompt_parts.append("   - Microsoft JDBC driver testing strategies")
        prompt_parts.append("   - SQL Server connectivity validation")
        prompt_parts.append("   - Performance benchmarking approaches")
        prompt_parts.append("   - Monitoring and alerting setup")
        prompt_parts.append("")
        
        prompt_parts.append("**EXPERT OUTPUT REQUIREMENTS:**")
        prompt_parts.append("- COMPREHENSIVE MSSQL JDBC EXPERT RESPONSE (aim for 4500+ characters)")
        prompt_parts.append("- Reference specific JDBC issue numbers and commit SHAs from analysis above")
        prompt_parts.append("- Include multiple practical Microsoft JDBC driver code examples")
        prompt_parts.append("- Provide expert-level implementation guidance with exact parameters")
        prompt_parts.append("- Connect solution to Microsoft JDBC driver patterns and SQL Server best practices")
        prompt_parts.append("- Each section should demonstrate deep JDBC driver expertise and be immediately actionable")
        prompt_parts.append("- Focus on Microsoft-specific implementations and proven solutions from the JDBC repository")
        
        return "\n".join(prompt_parts)

    def _generate_enhanced_fallback_response(self, issue_data: Dict[str, Any], repo_context: Dict[str, Any]) -> str:
        """Generate enhanced Microsoft MSSQL JDBC-focused fallback response with repository context"""
        
        related_jdbc_count = len(repo_context.get('related_jdbc_issues', []))
        mssql_commits_count = len(repo_context.get('mssql_jdbc_commits', []))
        mssql_open_count = len(repo_context.get('mssql_jdbc_open_issues', []))
        mssql_closed_count = len(repo_context.get('mssql_jdbc_closed_issues', []))
        is_mssql_jdbc = repo_context.get('is_mssql_jdbc_repo', False)
        
        # Get some specific issue details for better context
        issue_title = issue_data.get('issue_title', 'Unknown Issue')
        issue_body = issue_data.get('issue_body', '')
        repo_name = issue_data.get('repo_name', 'Unknown')
        
        # Extract relevant JDBC issues for reference
        related_issues_text = ""
        if repo_context.get('related_jdbc_issues'):
            related_issues_text = "\n**🔗 Related Microsoft MSSQL JDBC Issues:**\n"
            for issue in repo_context['related_jdbc_issues'][:3]:
                state = "✅ RESOLVED" if issue['state'] == 'closed' else "🔴 OPEN"
                related_issues_text += f"- {state} #{issue['number']}: {issue['title'][:80]}...\n"
                if issue['state'] == 'closed':
                    related_issues_text += f"  💡 *This was successfully resolved - check solution patterns*\n"
        
        return f"""**🔧 Microsoft SQL Server JDBC Driver Expert Analysis**

**Repository Context:** {repo_name}
**Issue:** #{issue_data.get('issue_number', 'N/A')} - {issue_title}
**MSSQL JDBC Intelligence:** {'DIRECT ANALYSIS' if is_mssql_jdbc else 'CROSS-REFERENCE EXPERTISE'}

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

*This analysis leverages expertise from {related_jdbc_count} related Microsoft MSSQL JDBC issues and {mssql_commits_count} recent driver development activities.*

---
**Generated by Git.AI Enhanced Assistant with Microsoft SQL Server JDBC Specialization**
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
