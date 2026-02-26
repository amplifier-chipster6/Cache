"""OpenAI embeddings client with retry on 429/5xx.

API key is read exclusively from the OPENAI_API_KEY environment variable.
No key material is ever logged.
"""

import logging
import os
import time

import openai

_LOG = logging.getLogger(__name__)
_RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})


class OpenAIEmbedderClient:
    """Thin wrapper around openai.OpenAI for creating text embeddings."""

    def __init__(self, *, timeout_s: float = 10.0):
        api_key = os.environ.get("OPENAI_API_KEY")
        self._client = openai.OpenAI(api_key=api_key, timeout=timeout_s)

    def embed_one(
        self,
        text: str,
        *,
        model: str,
        dimensions: int,
        max_retries: int = 3,
        backoff_seq: list[float] | None = None,
    ) -> list[float]:
        """Return embedding vector for *text*.

        Retries on 429 and 5xx up to *max_retries* times using *backoff_seq*
        (seconds between successive attempts).  Raises on exhaustion or any
        non-retryable status.
        """
        if backoff_seq is None:
            backoff_seq = [1.0, 2.0, 4.0]

        last_exc: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                response = self._client.embeddings.create(
                    model=model,
                    input=[text],
                    dimensions=dimensions,
                )
                embedding: list[float] = response.data[0].embedding
                if len(embedding) != dimensions:
                    raise ValueError(
                        f"Embedder returned {len(embedding)} dims; expected {dimensions}"
                    )
                return embedding
            except openai.RateLimitError as exc:
                last_exc = exc
                _LOG.warning("embed_one: rate-limited (attempt %d/%d)", attempt, max_retries)
            except openai.APIStatusError as exc:
                if exc.status_code in _RETRYABLE_STATUSES:
                    last_exc = exc
                    _LOG.warning(
                        "embed_one: status %s (attempt %d/%d)",
                        exc.status_code,
                        attempt,
                        max_retries,
                    )
                else:
                    raise

            # Sleep before next attempt (skip sleep after the final attempt)
            if attempt < max_retries:
                delay = backoff_seq[min(attempt, len(backoff_seq) - 1)] if backoff_seq else 0.0
                time.sleep(delay)

        assert last_exc is not None
        raise last_exc
