import os
import re
import json
import logging

import requests

from app.candisift.domain import ports

log = logging.getLogger("candisift.github")

class GitHubEnricherAdapter:
    def __init__(self, llm_provider: ports.LLMProvider, default_model: str) -> None:
        self.llm = llm_provider
        self.model = default_model

    def _fetch_github_api(self, api_url: str, params: dict | None = None) -> tuple[int, dict | list]:
        headers = {}
        github_token = os.environ.get("GITHUB_TOKEN")
        if github_token:
            headers["Authorization"] = f"token {github_token}"

        response = requests.get(api_url, params=params, timeout=10, headers=headers)
        status_code = response.status_code

        rate_limit_remaining = response.headers.get("X-RateLimit-Remaining")
        rate_limit_limit = response.headers.get("X-RateLimit-Limit")
        if rate_limit_remaining is not None and rate_limit_limit is not None:
            remaining = int(rate_limit_remaining)
            limit = int(rate_limit_limit)
            if remaining < 10:
                log.warning(f"GitHub API rate limit low: {remaining}/{limit} requests remaining.")

        data = response.json() if status_code == 200 else {}
        return status_code, data

    def _extract_github_username(self, github_url: str) -> str | None:
        if not github_url:
            return None
        github_url = github_url.replace(" ", "").strip()
        patterns = [
            r"https?://(?:www\.)?github\.com/([^/]+)",
            r"github\.com/([^/]+)",
            r"@([^/]+)",
            r"^([a-zA-Z0-9-]+)$",
        ]
        for pattern in patterns:
            match = re.search(pattern, github_url)
            if match:
                username = match.group(1)
                if "?" in username:
                    username = username.split("?", 1)[0]
                return username
        return None

    def _fetch_contributions_count(self, owner: str, contributors_data: list) -> tuple[int, int]:
        user_contributions = 0
        total_contributions = 0
        for contributor in contributors_data:
            if isinstance(contributor, dict):
                contributions = contributor.get("contributions", 0)
                total_contributions += contributions
                if contributor.get("login", "").lower() == owner.lower():
                    user_contributions = contributions
        return user_contributions, total_contributions

    def _fetch_repo_contributors(self, owner: str, repo_name: str) -> list[dict]:
        api_url = f"https://api.github.com/repos/{owner}/{repo_name}/contributors"
        status_code, contributors_data = self._fetch_github_api(api_url)
        if status_code == 200 and isinstance(contributors_data, list):
            return contributors_data
        return []

    def enrich(self, github_url: str) -> list[dict]:
        username = self._extract_github_username(github_url)
        if not username:
            log.warning(f"Could not extract username from: {github_url}")
            return []

        api_url = f"https://api.github.com/users/{username}/repos"
        # Limit to 30 repos to reduce API calls (each repo needs a /contributors call)
        params = {"sort": "pushed", "per_page": 30, "type": "owner"}
        status_code, repos_data = self._fetch_github_api(api_url, params=params)

        if status_code != 200 or not isinstance(repos_data, list):
            log.warning(f"Failed to fetch repositories for {username}. Status: {status_code}")
            return []

        projects = []
        for repo in repos_data:
            if repo.get("fork") and repo.get("forks_count", 0) < 5:
                continue

            repo_name = repo.get("name")
            contributors_data = self._fetch_repo_contributors(username, repo_name)
            contributor_count = len(contributors_data)
            user_contributions, total_contributions = self._fetch_contributions_count(username, contributors_data)

            project_type = "open_source" if contributor_count > 1 else "self_project"

            project = {
                "name": repo.get("name"),
                "description": repo.get("description"),
                "github_url": repo.get("html_url"),
                "live_url": repo.get("homepage") if repo.get("homepage") else None,
                "technologies": [repo.get("language")] if repo.get("language") else [],
                "project_type": project_type,
                "contributor_count": contributor_count,
                "author_commit_count": user_contributions,
                "total_commit_count": total_contributions,
                "github_details": {
                    "stars": repo.get("stargazers_count", 0),
                    "forks": repo.get("forks_count", 0),
                    "language": repo.get("language"),
                    "topics": repo.get("topics", []),
                    "contributors": contributor_count,
                },
            }
            projects.append(project)

        projects.sort(key=lambda x: x["github_details"]["stars"], reverse=True)

        if not projects:
            return []

        # Prepare for LLM Selection
        projects_data = []
        for project in projects:
            if project.get("author_commit_count") == 0:
                continue
            projects_data.append({
                "name": project.get("name"),
                "description": project.get("description"),
                "github_url": project.get("github_url"),
                "live_url": project.get("live_url"),
                "technologies": project.get("technologies", []),
                "project_type": project.get("project_type", "self_project"),
                "contributor_count": project.get("contributor_count", 1),
                "author_commit_count": project.get("author_commit_count", 0),
                "total_commit_count": project.get("total_commit_count", 0),
            })

        projects_json = json.dumps(projects_data, indent=2)

        try:
            selector = self.llm.github_selector(self.model)
            unique_projects = selector.select(projects_json)
            if unique_projects and len(unique_projects) > 0:
                return unique_projects
        except Exception as e:
            log.error(f"Error using LLM for project selection: {e}")

        # Fallback to top 7
        return projects_data[:7]
