"""FastAPI gateway: webhooks, normalization, routing, platform clients, dispatch."""

from __future__ import annotations

import logging
import os
import re
from contextlib import asynccontextmanager
from typing import Any

import httpx
import yaml
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Request
from pydantic import BaseModel, Field

from skills import SKILL_REGISTRY, format_response

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

CONFIG: dict[str, Any] = {}


def _interpolate_env(obj: Any) -> Any:
    """Recursively substitute ``${VAR_NAME}`` placeholders in parsed YAML."""
    if isinstance(obj, str):
        return re.sub(
            r"\$\{(\w+)\}",
            lambda m: os.environ.get(m.group(1), m.group(0)),
            obj,
        )
    if isinstance(obj, dict):
        return {k: _interpolate_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_interpolate_env(i) for i in obj]
    return obj


def load_config() -> dict[str, Any]:
    global CONFIG
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    CONFIG = _interpolate_env(raw) or {}
    return CONFIG


load_config()

# ---------------------------------------------------------------------------
# NormalizedEvent model
# ---------------------------------------------------------------------------


class NormalizedEvent(BaseModel):
    platform: str  # "jira" | "gitlab"
    project: str
    issue_id: str
    issue_type: str  # "issue" | "task" | "bug" | "merge_request" | …
    issue_title: str
    issue_description: str | None = None
    comment_body: str
    comment_author: str
    mention: str = ""
    mention_body: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    raw_payload: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Platform clients
# ---------------------------------------------------------------------------

CLIENTS: dict[str, Any] = {}


class JiraClient:
    def __init__(self) -> None:
        email = os.environ["JIRA_USER_EMAIL"]
        token = os.environ["JIRA_API_TOKEN"]
        base_url = os.environ["JIRA_BASE_URL"].rstrip("/")
        self.http = httpx.AsyncClient(
            base_url=base_url,
            auth=(email, token),
            headers={"Content-Type": "application/json"},
            timeout=30,
        )

    async def post_comment(self, event: NormalizedEvent, body: str) -> None:
        url = f"/rest/api/2/issue/{event.issue_id}/comment"
        resp = await self.http.post(url, json={"body": body})
        resp.raise_for_status()

    async def fetch_context(self, event: NormalizedEvent) -> dict[str, Any]:
        url = f"/rest/api/2/issue/{event.issue_id}?fields=comment,description,summary"
        resp = await self.http.get(url)
        resp.raise_for_status()
        data = resp.json()
        fields = data.get("fields", {})
        comments_raw = fields.get("comment", {}).get("comments", [])
        comments = [
            {
                "author": c.get("author", {}).get("displayName", "Unknown"),
                "body": c.get("body", ""),
                "created": c.get("created", ""),
            }
            for c in comments_raw
        ]
        return {
            "summary": fields.get("summary", ""),
            "description": fields.get("description", ""),
            "comments": comments,
        }

    async def close(self) -> None:
        await self.http.aclose()


class GitLabClient:
    def __init__(self) -> None:
        base_url = os.environ["GITLAB_BASE_URL"].rstrip("/")
        token = os.environ["GITLAB_API_TOKEN"]
        self.http = httpx.AsyncClient(
            base_url=f"{base_url}/api/v4",
            headers={
                "PRIVATE-TOKEN": token,
                "Content-Type": "application/json",
            },
            timeout=30,
        )

    def _noteable_path(self, event: NormalizedEvent) -> str:
        pid = event.metadata["project_id"]
        if event.issue_type == "merge_request":
            return f"/projects/{pid}/merge_requests/{event.issue_id}"
        return f"/projects/{pid}/issues/{event.issue_id}"

    async def post_comment(self, event: NormalizedEvent, body: str) -> None:
        path = self._noteable_path(event)
        resp = await self.http.post(f"{path}/notes", json={"body": body})
        resp.raise_for_status()

    async def fetch_context(self, event: NormalizedEvent) -> dict[str, Any]:
        path = self._noteable_path(event)
        result: dict[str, Any] = {}

        # Fetch the issue/MR itself
        resp = await self.http.get(path)
        resp.raise_for_status()
        data = resp.json()
        result["summary"] = data.get("title", "")
        result["description"] = data.get("description", "")

        # Fetch notes (comments)
        resp = await self.http.get(f"{path}/notes", params={"per_page": 100, "sort": "asc"})
        resp.raise_for_status()
        notes = resp.json()
        result["comments"] = [
            {
                "author": n.get("author", {}).get("username", "Unknown"),
                "body": n.get("body", ""),
                "created": n.get("created_at", ""),
            }
            for n in notes
        ]

        # For MRs, also fetch the diff
        if event.issue_type == "merge_request":
            resp = await self.http.get(f"{path}/changes")
            resp.raise_for_status()
            changes = resp.json().get("changes", [])
            diff_parts = []
            for change in changes:
                header = f"diff --git a/{change['old_path']} b/{change['new_path']}"
                diff_parts.append(f"{header}\n{change.get('diff', '')}")
            result["diff"] = "\n".join(diff_parts)

        return result

    async def close(self) -> None:
        await self.http.aclose()


# Conditionally instantiate clients
if os.environ.get("JIRA_BASE_URL") and os.environ.get("JIRA_API_TOKEN"):
    CLIENTS["jira"] = JiraClient()

if os.environ.get("GITLAB_BASE_URL") and os.environ.get("GITLAB_API_TOKEN"):
    CLIENTS["gitlab"] = GitLabClient()


# ---------------------------------------------------------------------------
# Parsing functions
# ---------------------------------------------------------------------------


def _extract_adf_text(node: dict | str | list) -> str:
    """Recursively extract plain text from Atlassian Document Format nodes."""
    if isinstance(node, str):
        return node
    if isinstance(node, list):
        return "".join(_extract_adf_text(n) for n in node)
    if isinstance(node, dict):
        if node.get("type") == "text":
            return node.get("text", "")
        if node.get("type") == "mention":
            return node.get("attrs", {}).get("text", "")
        # Recurse into content
        children = node.get("content", [])
        parts = [_extract_adf_text(c) for c in children]
        # Add newlines for block-level elements
        if node.get("type") in ("paragraph", "heading", "bulletList", "orderedList", "listItem"):
            return "".join(parts) + "\n"
        return "".join(parts)
    return ""


def parse_jira_event(payload: dict[str, Any]) -> NormalizedEvent | None:
    """Parse a Jira webhook payload into a NormalizedEvent."""
    if payload.get("webhookEvent") != "comment_created":
        return None

    issue = payload.get("issue", {})
    comment = payload.get("comment", {})
    fields = issue.get("fields", {})

    # Handle comment body — may be a string or ADF object
    comment_body = comment.get("body", "")
    if isinstance(comment_body, dict):
        comment_body = _extract_adf_text(comment_body).strip()

    issue_type_raw = fields.get("issuetype", {}).get("name", "").lower()
    project_key = fields.get("project", {}).get("key", "")

    return NormalizedEvent(
        platform="jira",
        project=project_key,
        issue_id=issue.get("key", ""),
        issue_type=issue_type_raw,
        issue_title=fields.get("summary", ""),
        issue_description=fields.get("description", ""),
        comment_body=comment_body,
        comment_author=comment.get("author", {}).get("displayName", "unknown"),
        raw_payload=payload,
    )


def parse_gitlab_event(payload: dict[str, Any]) -> NormalizedEvent | None:
    """Parse a GitLab webhook payload into a NormalizedEvent."""
    if payload.get("object_kind") != "note":
        return None

    attrs = payload.get("object_attributes", {})
    noteable_type = attrs.get("noteable_type", "")

    if noteable_type == "Issue":
        issue_data = payload.get("issue", {})
        issue_type = "issue"
    elif noteable_type == "MergeRequest":
        issue_data = payload.get("merge_request", {})
        issue_type = "merge_request"
    else:
        return None

    project = payload.get("project", {})

    return NormalizedEvent(
        platform="gitlab",
        project=project.get("path_with_namespace", ""),
        issue_id=str(issue_data.get("iid", "")),
        issue_type=issue_type,
        issue_title=issue_data.get("title", ""),
        issue_description=issue_data.get("description", ""),
        comment_body=attrs.get("note", ""),
        comment_author=payload.get("user", {}).get("username", "unknown"),
        metadata={"project_id": project.get("id")},
        raw_payload=payload,
    )


# ---------------------------------------------------------------------------
# Route function
# ---------------------------------------------------------------------------


def route(event: NormalizedEvent) -> tuple[str, str] | None:
    """Return the first ``(skill_name, mention_body)`` found, or *None*."""
    bot_accounts = CONFIG.get("gateway", {}).get("bot_accounts", [])
    if event.comment_author in bot_accounts:
        return None

    for skill_name, skill_cfg in CONFIG.get("skills", {}).items():
        mention = skill_cfg.get("mention_name", skill_name)
        m = re.search(rf"@{re.escape(mention)}\b\s*(.*)", event.comment_body, re.DOTALL)
        if m:
            return skill_name, m.group(1).strip()

    return None


# ---------------------------------------------------------------------------
# Dispatch function
# ---------------------------------------------------------------------------


async def dispatch(event: NormalizedEvent, skill_name: str, mention_body: str) -> None:
    """Instantiate a skill, execute it, and post the response."""
    client = CLIENTS.get(event.platform)
    if not client:
        logger.error("No client configured for platform %s", event.platform)
        return

    skill_cls = SKILL_REGISTRY.get(skill_name)
    if not skill_cls:
        logger.error("Unknown skill: %s", skill_name)
        return

    # Build skill config from global config
    skill_config = dict(CONFIG.get("skills", {}).get(skill_name, {}))
    skill_config["_client"] = client

    # Inject mention info into event
    mention_name = skill_config.get("mention_name", skill_name)
    event = event.model_copy(update={"mention": mention_name, "mention_body": mention_body})

    try:
        skill = skill_cls()
        body = await skill.execute(event, skill_config)
        await client.post_comment(event, body)

    except Exception:
        logger.exception("Skill %s failed for event %s/%s", skill_name, event.platform, event.issue_id)
        try:
            error_body = format_response(
                skill_name,
                "An error occurred while processing your request. Please try again or contact an admin.",
            )
            await client.post_comment(event, error_body)
        except Exception:
            logger.exception("Failed to post error comment")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=getattr(logging, log_level, logging.INFO))
    logger.info("Gateway starting, registered skills: %s", list(SKILL_REGISTRY.keys()))
    yield
    # Shutdown: close all httpx clients
    for name, client in CLIENTS.items():
        logger.info("Closing %s client", name)
        await client.close()


app = FastAPI(title="Mention Hook Gateway", lifespan=lifespan)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "skills": list(SKILL_REGISTRY.keys()),
        "platforms": list(CLIENTS.keys()),
    }


async def _handle_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    parser,
):
    payload = await request.json()
    event = parser(payload)
    if not event:
        return {"status": "ignored", "reason": "unsupported event type"}

    match = route(event)
    if not match:
        return {"status": "ignored", "reason": "no skill mention found"}

    skill_name, mention_body = match
    background_tasks.add_task(dispatch, event, skill_name, mention_body)

    return {"status": "accepted", "skill": skill_name}


@app.post("/webhook/jira")
async def webhook_jira(request: Request, background_tasks: BackgroundTasks):
    return await _handle_webhook(request, background_tasks, parse_jira_event)


@app.post("/webhook/gitlab")
async def webhook_gitlab(request: Request, background_tasks: BackgroundTasks):
    return await _handle_webhook(request, background_tasks, parse_gitlab_event)
