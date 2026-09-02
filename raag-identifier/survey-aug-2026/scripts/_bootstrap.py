"""Put survey-aug-2026/ on sys.path so `from common import ...` works from anywhere.

Every script in this folder starts with `import _bootstrap  # noqa: F401`. That is the whole
mechanism -- no package installs, no PYTHONPATH to remember, and the scripts stay runnable
as plain files from any working directory.
"""

import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)
