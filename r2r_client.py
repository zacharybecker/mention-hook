"""R2R async client wrapper — singleton with admin login."""

from __future__ import annotations

import logging
import os

from r2r import R2RAsyncClient

logger = logging.getLogger(__name__)

_client: R2RAsyncClient | None = None
_logged_in: bool = False


async def _get_client() -> R2RAsyncClient:
    """Return a module-level singleton R2RAsyncClient, logged in as admin."""
    global _client, _logged_in

    if _client is None:
        _client = R2RAsyncClient(
            base_url=os.environ["R2R_BASE_URL"],
        )

    if not _logged_in:
        email = os.environ["R2R_ADMIN_EMAIL"]
        password = os.environ["R2R_ADMIN_PASSWORD"]
        await _client.users.login(email, password)
        _logged_in = True
        logger.info("R2R admin login successful")

    return _client


async def search(query: str, limit: int = 5) -> list[dict]:
    """Search the R2R knowledge base and return result chunks.

    Returns a list of dicts with ``text`` and ``source`` keys.
    """
    client = await _get_client()
    raw = await client.retrieval.search(query=query, limit=limit)

    # Normalise into a simple list regardless of SDK response shape.
    results: list[dict] = []
    chunks = raw if isinstance(raw, list) else getattr(raw, "results", None) or []
    for i, chunk in enumerate(chunks, 1):
        if isinstance(chunk, dict):
            text = chunk.get("text", chunk.get("content", ""))
            source = chunk.get("metadata", {}).get("title", f"chunk-{i}")
        else:
            # SDK may return objects with attributes instead of dicts.
            text = getattr(chunk, "text", "") or getattr(chunk, "content", "")
            meta = getattr(chunk, "metadata", {}) or {}
            source = (meta.get("title") if isinstance(meta, dict) else getattr(meta, "title", None)) or f"chunk-{i}"
        results.append({"text": text, "source": source})

    return results
