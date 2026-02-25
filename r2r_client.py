"""R2R async client wrapper — singleton with admin login and token refresh."""

from __future__ import annotations

import logging
import os
import time

from r2r import R2RAsyncClient

logger = logging.getLogger(__name__)

_client: R2RAsyncClient | None = None
_login_time: float = 0
_LOGIN_TTL: int = int(os.environ.get("R2R_LOGIN_TTL", "3600"))


async def _get_client() -> R2RAsyncClient:
    """Return a module-level singleton R2RAsyncClient, logged in as admin."""
    global _client, _login_time

    if _client is None:
        _client = R2RAsyncClient(
            base_url=os.environ["R2R_BASE_URL"],
        )

    if _login_time == 0 or (time.monotonic() - _login_time) >= _LOGIN_TTL:
        await _login()

    return _client


async def _login() -> None:
    """Authenticate with R2R and record the login timestamp."""
    global _login_time
    email = os.environ["R2R_ADMIN_EMAIL"]
    password = os.environ["R2R_ADMIN_PASSWORD"]
    await _client.users.login(email, password)
    _login_time = time.monotonic()
    logger.info("R2R admin login successful")


async def search(query: str, limit: int = 5) -> list[dict]:
    """Search the R2R knowledge base and return result chunks.

    Returns a list of dicts with ``text`` and ``source`` keys.
    """
    client = await _get_client()
    try:
        raw = await client.retrieval.search(query=query, limit=limit)
    except Exception:
        # Token may have been revoked server-side — re-login and retry once.
        logger.warning("R2R search failed, re-authenticating and retrying")
        await _login()
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
