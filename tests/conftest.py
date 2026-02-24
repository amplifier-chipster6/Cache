import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
module_path = root / "modules" / "cache-normalizer"
sys.path.insert(0, str(module_path))
