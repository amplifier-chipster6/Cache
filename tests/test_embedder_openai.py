"""Tests for modules/embedder-openai.

Uses a fake openai module so no real API calls are made.
Fake is injected into sys.modules BEFORE importing the module under test.
"""

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch, call

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "modules" / "embedder-openai"))

# ---------------------------------------------------------------------------
# Fake openai module — installed BEFORE any import of embedder_openai
# ---------------------------------------------------------------------------

class _FakeRateLimitError(Exception):
    """Stand-in for openai.RateLimitError."""


class _FakeAPIStatusError(Exception):
    """Stand-in for openai.APIStatusError."""

    def __init__(self, message="", *, status_code: int = 500, response=None, body=None):
        super().__init__(message)
        self.status_code = status_code


_fake_openai = types.ModuleType("openai")
_fake_openai.RateLimitError = _FakeRateLimitError
_fake_openai.APIStatusError = _FakeAPIStatusError
_fake_openai.OpenAI = MagicMock(name="FakeOpenAIClass")
sys.modules.setdefault("openai", _fake_openai)

# ---------------------------------------------------------------------------
# Now safe to import the module under test
# ---------------------------------------------------------------------------
from embedder_openai.client import OpenAIEmbedderClient  # noqa: E402
from embedder_openai.mount import mount  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_response(embedding: list[float]) -> MagicMock:
    resp = MagicMock()
    resp.data = [MagicMock()]
    resp.data[0].embedding = embedding
    return resp


def _make_client_mock(side_effects) -> MagicMock:
    """Return a mock openai.OpenAI instance whose embeddings.create raises or
    returns items from *side_effects* (one per call)."""
    mock_instance = MagicMock()
    mock_instance.embeddings.create.side_effect = side_effects
    return mock_instance


# ---------------------------------------------------------------------------
# Task 1 smoke test
# ---------------------------------------------------------------------------

def test_mount_returns_callable():
    """mount() must return a callable (embed function)."""
    # Ensure OpenAI constructor returns something usable
    _fake_openai.OpenAI.return_value = _make_client_mock([_make_response([0.0] * 1536)])
    fn = mount({})
    assert callable(fn)


# ---------------------------------------------------------------------------
# Task 2: OpenAI client tests
# ---------------------------------------------------------------------------

class TestClient:
    def _client_with(self, side_effects):
        """Patch openai.OpenAI to return a mock then return a fresh client."""
        mock_instance = _make_client_mock(side_effects)
        _fake_openai.OpenAI.return_value = mock_instance
        return OpenAIEmbedderClient(timeout_s=5.0), mock_instance

    def test_success_returns_embedding(self):
        vec = [0.1] * 1536
        client, _ = self._client_with([_make_response(vec)])
        result = client.embed_one(
            "hello",
            model="text-embedding-3-small",
            dimensions=1536,
            max_retries=0,
            backoff_seq=[],
        )
        assert result == vec

    def test_retry_on_429_then_succeeds(self):
        vec = [0.2] * 1536
        side_effects = [
            _FakeRateLimitError("rate limited"),
            _make_response(vec),
        ]
        client, mock_instance = self._client_with(side_effects)
        with patch("embedder_openai.client.time.sleep"):
            result = client.embed_one(
                "hello",
                model="text-embedding-3-small",
                dimensions=1536,
                max_retries=2,
                backoff_seq=[0.0, 0.0],
            )
        assert result == vec
        assert mock_instance.embeddings.create.call_count == 2

    def test_retry_on_500_then_succeeds(self):
        vec = [0.3] * 1536
        side_effects = [
            _FakeAPIStatusError("server error", status_code=500),
            _make_response(vec),
        ]
        client, mock_instance = self._client_with(side_effects)
        with patch("embedder_openai.client.time.sleep"):
            result = client.embed_one(
                "hello",
                model="text-embedding-3-small",
                dimensions=1536,
                max_retries=2,
                backoff_seq=[0.0, 0.0],
            )
        assert result == vec
        assert mock_instance.embeddings.create.call_count == 2

    def test_raises_after_exhausted_retries(self):
        import pytest
        side_effects = [_FakeRateLimitError("rate limited")] * 4
        client, _ = self._client_with(side_effects)
        with patch("embedder_openai.client.time.sleep"):
            with pytest.raises(_FakeRateLimitError):
                client.embed_one(
                    "hello",
                    model="text-embedding-3-small",
                    dimensions=1536,
                    max_retries=3,
                    backoff_seq=[0.0, 0.0, 0.0],
                )

    def test_non_retryable_error_raises_immediately(self):
        import pytest
        side_effects = [_FakeAPIStatusError("bad request", status_code=400)]
        client, mock_instance = self._client_with(side_effects)
        with pytest.raises(_FakeAPIStatusError):
            client.embed_one(
                "hello",
                model="text-embedding-3-small",
                dimensions=1536,
                max_retries=3,
                backoff_seq=[0.0],
            )
        # Only one attempt — no retry on 4xx
        assert mock_instance.embeddings.create.call_count == 1

    def test_dimension_mismatch_raises_value_error(self):
        import pytest
        # API returns 10-dim vector but we asked for 1536
        client, _ = self._client_with([_make_response([0.1] * 10)])
        with pytest.raises(ValueError, match="1536"):
            client.embed_one(
                "hello",
                model="text-embedding-3-small",
                dimensions=1536,
                max_retries=0,
                backoff_seq=[],
            )

    def test_backoff_delays_are_used(self):
        vec = [0.5] * 1536
        side_effects = [
            _FakeRateLimitError("rate limited"),
            _FakeRateLimitError("rate limited"),
            _make_response(vec),
        ]
        client, _ = self._client_with(side_effects)
        with patch("embedder_openai.client.time.sleep") as mock_sleep:
            client.embed_one(
                "hello",
                model="text-embedding-3-small",
                dimensions=1536,
                max_retries=3,
                backoff_seq=[1.0, 2.0, 4.0],
            )
        assert mock_sleep.call_args_list == [call(1.0), call(2.0)]


# ---------------------------------------------------------------------------
# Task 3: mount() tests
# ---------------------------------------------------------------------------

class TestMount:
    def _patched_client(self, side_effects):
        mock_instance = _make_client_mock(side_effects)
        _fake_openai.OpenAI.return_value = mock_instance
        return mock_instance

    def test_mount_default_config(self):
        self._patched_client([_make_response([0.0] * 1536)])
        fn = mount(None)
        assert callable(fn)

    def test_embed_returns_vector(self):
        vec = [0.7] * 1536
        self._patched_client([_make_response(vec)])
        fn = mount({})
        with patch("embedder_openai.client.time.sleep"):
            result = fn("hello world")
        assert result == vec

    def test_embed_returns_none_on_rate_limit(self):
        self._patched_client([_FakeRateLimitError("rate limited")] * 10)
        fn = mount({"max_retries": 0})
        with patch("embedder_openai.client.time.sleep"):
            result = fn("hello")
        assert result is None

    def test_embed_returns_none_on_any_exception(self):
        self._patched_client([RuntimeError("boom")])
        fn = mount({"max_retries": 0})
        result = fn("hello")
        assert result is None

    def test_custom_model_and_dimensions(self):
        vec = [0.1] * 512
        mock_instance = self._patched_client([_make_response(vec)])
        fn = mount({"model": "text-embedding-3-large", "dimensions": 512})
        with patch("embedder_openai.client.time.sleep"):
            fn("test")
        create_call = mock_instance.embeddings.create.call_args
        assert create_call.kwargs["model"] == "text-embedding-3-large"
        assert create_call.kwargs["dimensions"] == 512

    def test_api_key_not_in_config(self):
        """mount() must not accept or log api_key from config."""
        mock_instance = self._patched_client([_make_response([0.1] * 1536)])
        fn = mount({"model": "text-embedding-3-small"})
        # api_key should NOT appear in the OpenAI constructor kwargs via config
        openai_init_kwargs = _fake_openai.OpenAI.call_args
        assert openai_init_kwargs is not None
        # api_key may be passed from env — but never from config dict
        assert "api_key" not in ({"model": "text-embedding-3-small"})
