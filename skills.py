"""Skill definitions, LLM utilities, and the skill registry."""

from __future__ import annotations

import logging
import os
import re
from abc import ABC, abstractmethod
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

import r2r_client

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts"

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



def format_response(skill_name: str, body: str) -> str:
    """Prefix a skill response with a bold @mention header."""
    return f"**@{skill_name}**\n\n{body}"


# ---------------------------------------------------------------------------
# Base skill
# ---------------------------------------------------------------------------


class BaseSkill(ABC):
    @abstractmethod
    async def execute(self, event: Any, config: dict[str, Any]) -> str:
        ...


# ---------------------------------------------------------------------------
# Skill implementations
# ---------------------------------------------------------------------------


class R2RSkill(BaseSkill):
    """Generic R2R knowledge-base skill.

    Searches R2R for context, then sends it with the user's question to the
    LLM.  Uses ``event.mention`` to select the prompt file and response
    header, so the same class can be registered under multiple names
    (e.g. ``support``, ``it-support``, ``project-support``).
    """

    async def execute(self, event: Any, config: dict[str, Any]) -> str:
        skill_name = event.mention
        question = event.mention_body
        results = await r2r_client.search(
            query=question,
            limit=config.get("r2r_search_limit", 5),
        )

        # Build context from returned chunks
        context_parts: list[str] = []
        for i, chunk in enumerate(results, 1):
            context_parts.append(f"[{i}] Source: {chunk['source']}\n{chunk['text']}")

        context = "\n\n---\n\n".join(context_parts)
        context = truncate(context, config.get("max_context_chars", 16000))

        system_prompt = load_prompt(skill_name)
        user_msg = f"## Context\n\n{context}\n\n## Question\n\n{question}"

        body = await call_llm(system_prompt, user_msg, config["model"])
        return format_response(skill_name, body)


class ReviewSkill(BaseSkill):
    """Review merge request diffs for issues."""

    async def execute(self, event: Any, config: dict[str, Any]) -> str:
        if event.issue_type != "merge_request":
            return format_response("review", "The review skill only works on merge requests.")

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
        return format_response("review", body)


class SummarizeSkill(BaseSkill):
    """Summarize an issue or MR discussion thread."""

    async def execute(self, event: Any, config: dict[str, Any]) -> str:
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
        return format_response("summarize", body)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SKILL_REGISTRY: dict[str, type[BaseSkill]] = {
    "support": R2RSkill,
    "review": ReviewSkill,
    "summarize": SummarizeSkill,
}
