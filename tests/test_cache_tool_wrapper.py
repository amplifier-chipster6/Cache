import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "modules" / "cache-tool-wrapper"))

from cache_tool_wrapper.tool import CacheToolWrapper  # noqa: E402


class DummyTool:
    name = "dummy"
    description = "dummy"

    def __init__(self):
        self.called = False

    async def execute(self, input):
        self.called = True
        return {"ok": True}


class DummyExactCache:
    def __init__(self, value):
        self.value = value
        self.last_key = None

    def get(self, key):
        self.last_key = key
        return self.value


def test_exact_cache_hit_tool():
    tool = DummyTool()
    exact_cache = DummyExactCache({"ok": "cached"})
    wrapper = CacheToolWrapper(tool, exact_cache=exact_cache, semantic_cache=None)

    result = asyncio.run(wrapper.execute({"input": True}))

    assert result == {"ok": "cached"}
    assert tool.called is False
    assert exact_cache.last_key == {"input": True}
