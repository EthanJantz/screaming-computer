"""Make the flat src/ modules and the vendored thinkdsp importable from tests."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # thinkdsp.py
sys.path.insert(0, str(ROOT / "src"))
