"""Skill definitions, LLM utilities, and the skill registry."""

from __future__ import annotations

import json
import logging
import os
import re
from abc import ABC, abstractmethod
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import httpx
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts"

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class SideEffect(BaseModel):
    action: str  # "add_labels" | "set_assignee" | "change_status" | "add_reaction"
    params: dict[str, Any]


class SkillResponse(BaseModel):
    body: str
    side_effects: list[SideEffect] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

_llm_client: AsyncOpenAI | None = None


def _get_llm_client() -> AsyncOpenAI:
    """Return a module-level lazy singleton AsyncOpenAI client."""
    global _llm_client
    if _llm_client is None:
        _llm_client = AsyncOpenAI(
            base_url=os.environ["LITELLM_BASE_URL"],
            api_key=os.environ.get("LITELLM_API_KEY", "no-key"),
        )
    return _llm_client


def load_prompt(skill_name: str) -> str:
    """Read prompts/{skill_name}.md and return its contents."""
    path = PROMPTS_DIR / f"{skill_name}.md"
    return path.read_text()


def truncate(text: str, max_chars: int, strategy: str = "end") -> str:
    """Truncate *text* to *max_chars*.

    Strategies:
        ``"end"``   — keep the first *max_chars* characters.
        ``"middle"`` — keep the first 40% and last 40% (useful for diffs
                       where headers and final hunks both matter).
    """
    if len(text) <= max_chars:
        return text

    marker = "\n\n... [truncated] ...\n\n"

    if strategy == "middle":
        keep = max_chars - len(marker)
        head = keep * 2 // 5
        tail = keep - head
        return text[:head] + marker + text[-tail:]

    return text[:max_chars - len(marker)] + marker


async def call_llm(
    system_prompt: str,
    user_message: str,
    model: str,
    max_tokens: int = 4096,
) -> str:
    """Send a chat completion request via LiteLLM and return the text."""
    client = _get_llm_client()
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


async def call_llm_json(
    system_prompt: str,
    user_message: str,
    model: str,
    max_tokens: int = 4096,
) -> Any:
    """Like :func:`call_llm` but parses the response as JSON."""
    system_prompt += "\n\nRespond with valid JSON only."
    raw = await call_llm(system_prompt, user_message, model, max_tokens)
    # Strip optional markdown code fences
    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", raw.strip())
    cleaned = re.sub(r"\n?```\s*$", "", cleaned)
    return json.loads(cleaned)


def format_response(skill_name: str, body: str) -> str:
    """Prefix a skill response with a bold @mention header."""
    return f"**@{skill_name}**\n\n{body}"


# ---------------------------------------------------------------------------
# Base skill
# ---------------------------------------------------------------------------


class BaseSkill(ABC):
    @abstractmethod
    async def execute(self, event: Any, config: dict[str, Any]) -> SkillResponse:
        ...


# ---------------------------------------------------------------------------
# Skill implementations
# ---------------------------------------------------------------------------


class SupportSkill(BaseSkill):
    """Answer questions using the R2R knowledge base."""

    async def execute(self, event: Any, config: dict[str, Any]) -> SkillResponse:
        question = event.mention_body
        r2r_base = os.environ["R2R_BASE_URL"]
        r2r_key = os.environ.get("R2R_API_KEY", "")

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{r2r_base}/v3/retrieval/search",
                headers={"Authorization": f"Bearer {r2r_key}"},
                json={
                    "query": question,
                    "limit": config.get("r2r_search_limit", 5),
                },
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])

        # Build context from returned chunks
        context_parts: list[str] = []
        for i, chunk in enumerate(results, 1):
            source = chunk.get("metadata", {}).get("title", f"chunk-{i}")
            text = chunk.get("text", chunk.get("content", ""))
            context_parts.append(f"[{i}] Source: {source}\n{text}")

        context = "\n\n---\n\n".join(context_parts)
        context = truncate(context, config.get("max_context_chars", 16000))

        system_prompt = load_prompt("support")
        user_msg = f"## Context\n\n{context}\n\n## Question\n\n{question}"

        body = await call_llm(system_prompt, user_msg, config["model"])
        return SkillResponse(body=format_response("support", body))


class ReviewSkill(BaseSkill):
    """Review merge request diffs for issues."""

    async def execute(self, event: Any, config: dict[str, Any]) -> SkillResponse:
        if event.issue_type != "merge_request":
            return SkillResponse(
                body=format_response(
                    "review",
                    "The review skill only works on merge requests.",
                )
            )

        client = config["_client"]
        context = await client.fetch_context(event)
        diff = context.get("diff", "")

        # Filter ignored files
        ignore_patterns = config.get("ignore_patterns", [])
        if ignore_patterns and diff:
            filtered_hunks: list[str] = []
            for hunk in re.split(r"(?=^diff --git )", diff, flags=re.MULTILINE):
                # Extract filename from diff header
                m = re.match(r"diff --git a/.+ b/(.+)", hunk)
                if m:
                    filename = m.group(1)
                    if any(fnmatch(filename, pat) for pat in ignore_patterns):
                        continue
                filtered_hunks.append(hunk)
            diff = "".join(filtered_hunks)

        diff = truncate(diff, config.get("max_diff_chars", 32000), strategy="middle")

        mr_description = event.issue_description or ""
        system_prompt = load_prompt("review")
        user_msg = f"## MR Description\n\n{mr_description}\n\n## Diff\n\n```diff\n{diff}\n```"

        body = await call_llm(system_prompt, user_msg, config["model"])
        return SkillResponse(body=format_response("review", body))


class SummarizeSkill(BaseSkill):
    """Summarize an issue or MR discussion thread."""

    async def execute(self, event: Any, config: dict[str, Any]) -> SkillResponse:
        client = config["_client"]
        context = await client.fetch_context(event)
        comments = context.get("comments", [])

        parts: list[str] = []
        # Prepend issue/MR title and description
        parts.append(f"# {event.issue_title}\n\n{event.issue_description or ''}")
        parts.append("---")

        for c in comments:
            author = c.get("author", "Unknown")
            timestamp = c.get("created", "")
            body = c.get("body", "")
            parts.append(f"**{author}** ({timestamp}):\n{body}\n---")

        thread = "\n\n".join(parts)
        thread = truncate(thread, config.get("max_context_chars", 24000))

        system_prompt = load_prompt("summarize")
        body = await call_llm(system_prompt, thread, config["model"])
        return SkillResponse(body=format_response("summarize", body))


class TriageSkill(BaseSkill):
    """Classify and triage an issue."""

    async def execute(self, event: Any, config: dict[str, Any]) -> SkillResponse:
        message = f"# {event.issue_title}\n\n{event.issue_description or ''}"

        system_prompt = load_prompt("triage")
        result = await call_llm_json(system_prompt, message, config["model"])

        priority = result.get("priority", "P3")
        labels = result.get("labels", [])
        assignee = result.get("assignee")
        reasoning = result.get("reasoning", "")

        # Validate against config
        valid_priorities = config.get("valid_priorities", ["P1", "P2", "P3", "P4"])
        if priority not in valid_priorities:
            priority = "P3"

        valid_labels = config.get("valid_labels", [])
        if valid_labels:
            labels = [l for l in labels if l in valid_labels]

        # Build side effects
        side_effects: list[SideEffect] = []
        if labels:
            side_effects.append(SideEffect(action="add_labels", params={"labels": labels}))
        if assignee:
            side_effects.append(SideEffect(action="set_assignee", params={"assignee": assignee}))

        body_lines = [
            f"**Priority:** {priority}",
            f"**Labels:** {', '.join(labels) if labels else 'none'}",
            f"**Assignee:** {assignee or 'unassigned'}",
            "",
            f"**Reasoning:** {reasoning}",
        ]

        return SkillResponse(
            body=format_response("triage", "\n".join(body_lines)),
            side_effects=side_effects,
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SKILL_REGISTRY: dict[str, type[BaseSkill]] = {
    "support": SupportSkill,
    "review": ReviewSkill,
    "summarize": SummarizeSkill,
    "triage": TriageSkill,
}
