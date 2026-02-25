"""FastAPI gateway: webhooks, normalization, routing, platform clients, dispatch."""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import re
import traceback
from contextlib import asynccontextmanager
from typing import Any

import httpx
import yaml
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Header, Request, Response
from pydantic import BaseModel, Field

from skills import SKILL_REGISTRY, SkillResponse, format_response

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

    async def apply_side_effect(self, event: NormalizedEvent, effect: Any) -> None:
        issue_url = f"/rest/api/2/issue/{event.issue_id}"
        if effect.action == "add_labels":
            labels = effect.params.get("labels", [])
            await self.http.put(
                issue_url,
                json={"update": {"labels": [{"add": l} for l in labels]}},
            )
        elif effect.action == "set_assignee":
            assignee = effect.params.get("assignee", "")
            await self.http.put(
                issue_url,
                json={"fields": {"assignee": {"name": assignee}}},
            )
        elif effect.action == "change_status":
            transition_id = effect.params.get("transition_id")
            if transition_id:
                await self.http.post(
                    f"{issue_url}/transitions",
                    json={"transition": {"id": transition_id}},
                )

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

    async def apply_side_effect(self, event: NormalizedEvent, effect: Any) -> None:
        path = self._noteable_path(event)
        if effect.action == "add_labels":
            labels = effect.params.get("labels", [])
            # Fetch existing labels, merge, and update
            resp = await self.http.get(path)
            resp.raise_for_status()
            existing = [l["name"] if isinstance(l, dict) else l for l in resp.json().get("labels", [])]
            merged = list(set(existing + labels))
            await self.http.put(path, json={"labels": ",".join(merged)})
        elif effect.action == "set_assignee":
            username = effect.params.get("assignee", "")
            # Resolve username to user ID
            resp = await self.http.get("/users", params={"username": username})
            resp.raise_for_status()
            users = resp.json()
            if users:
                user_id = users[0]["id"]
                await self.http.put(path, json={"assignee_ids": [user_id]})
        elif effect.action == "add_reaction":
            emoji = effect.params.get("emoji", "thumbsup")
            await self.http.post(f"{path}/award_emoji", json={"name": emoji})

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
# Webhook verification
# ---------------------------------------------------------------------------


def verify_jira_signature(body: bytes, signature_header: str | None) -> bool:
    secret = os.environ.get("JIRA_WEBHOOK_SECRET")
    if not secret:
        return True  # No secret configured, skip verification
    if not signature_header:
        return False
    expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature_header)


def verify_gitlab_token(token_header: str | None) -> bool:
    secret = os.environ.get("GITLAB_WEBHOOK_SECRET")
    if not secret:
        return True  # No secret configured, skip verification
    if not token_header:
        return False
    return hmac.compare_digest(secret, token_header)


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


def route(event: NormalizedEvent) -> list[dict[str, str]]:
    """Scan the comment body for @skill_name mentions and return matched skills."""
    # Skip comments from bot accounts
    bot_accounts = CONFIG.get("gateway", {}).get("bot_accounts", [])
    if event.comment_author in bot_accounts:
        return []

    skills_config = CONFIG.get("skills", {})
    matches: list[dict[str, str]] = []

    for skill_name in skills_config:
        pattern = rf"@{re.escape(skill_name)}\b(.*?)(?=@\w+|$)"
        for m in re.finditer(pattern, event.comment_body, re.DOTALL):
            mention_body = m.group(1).strip()
            matches.append({"name": skill_name, "body": mention_body})

    return matches


# ---------------------------------------------------------------------------
# Dispatch function
# ---------------------------------------------------------------------------


async def dispatch(event: NormalizedEvent, skill_name: str, mention_body: str) -> None:
    """Instantiate a skill, execute it, post the response, and apply side effects."""
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

    # Inject mention body into event
    event = event.model_copy(update={"mention": skill_name, "mention_body": mention_body})

    try:
        skill = skill_cls()
        result: SkillResponse = await skill.execute(event, skill_config)

        # Post the comment
        await client.post_comment(event, result.body)

        # Apply side effects
        for effect in result.side_effects:
            try:
                await client.apply_side_effect(event, effect)
            except Exception:
                logger.exception("Failed to apply side effect %s", effect.action)

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


@app.post("/webhook/jira")
async def webhook_jira(
    request: Request,
    background_tasks: BackgroundTasks,
    x_hub_signature: str | None = Header(None, alias="X-Hub-Signature"),
):
    body = await request.body()

    if not verify_jira_signature(body, x_hub_signature):
        return Response(status_code=401, content="Invalid signature")

    payload = await request.json()
    event = parse_jira_event(payload)
    if not event:
        return {"status": "ignored", "reason": "unsupported event type"}

    matches = route(event)
    if not matches:
        return {"status": "ignored", "reason": "no skill mentions found"}

    for match in matches:
        background_tasks.add_task(dispatch, event, match["name"], match["body"])

    return {"status": "accepted", "skills": [m["name"] for m in matches]}


@app.post("/webhook/gitlab")
async def webhook_gitlab(
    request: Request,
    background_tasks: BackgroundTasks,
    x_gitlab_token: str | None = Header(None, alias="X-Gitlab-Token"),
):
    if not verify_gitlab_token(x_gitlab_token):
        return Response(status_code=401, content="Invalid token")

    payload = await request.json()
    event = parse_gitlab_event(payload)
    if not event:
        return {"status": "ignored", "reason": "unsupported event type"}

    matches = route(event)
    if not matches:
        return {"status": "ignored", "reason": "no skill mentions found"}

    for match in matches:
        background_tasks.add_task(dispatch, event, match["name"], match["body"])

    return {"status": "accepted", "skills": [m["name"] for m in matches]}
