"""Mount function for the embedder-openai module.

Returns a callable ``embed(key: str) -> list[float] | None``.
Errors are caught and return ``None`` so the semantic path degrades
gracefully rather than surfacing exceptions to callers.
"""

import logging
from typing import Callable

from .client import OpenAIEmbedderClient

_LOG = logging.getLogger(__name__)

_DEFAULTS = {
    "model": "text-embedding-3-small",
    "dimensions": 1536,
    "batch_size": 32,
    "timeout_s": 10.0,
    "max_retries": 3,
    "retry_backoff_s": [1.0, 2.0, 4.0],
}


def mount(config: dict | None = None) -> Callable[[str], list[float] | None]:
    """Return an ``embed`` callable configured from *config*.

    Config keys (all optional, defaults used when absent):
    - model          str    openai model name
    - dimensions     int    embedding vector size
    - batch_size     int    reserved for future batching (unused now)
    - timeout_s      float  HTTP timeout in seconds
    - max_retries    int    retry attempts on 429/5xx
    - retry_backoff_s list  sleep seconds between retries

    No ``api_key`` field — key comes from ``OPENAI_API_KEY`` env var only.
    """
    cfg = {**_DEFAULTS, **(config or {})}

    client = OpenAIEmbedderClient(timeout_s=float(cfg["timeout_s"]))

    def embed(key: str) -> list[float] | None:
        try:
            return client.embed_one(
                key,
                model=cfg["model"],
                dimensions=int(cfg["dimensions"]),
                max_retries=int(cfg["max_retries"]),
                backoff_seq=list(cfg["retry_backoff_s"]),
            )
        except Exception:
            _LOG.warning("embedder-openai: embed failed, skipping semantic path")
            return None

    return embed
