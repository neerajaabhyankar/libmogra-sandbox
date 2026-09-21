"""Put `../raag-identifier/` on `sys.path` so `import utils` resolves. Import before `utils`."""

import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent / "raag-identifier")
if _root not in sys.path:
    sys.path.insert(0, _root)
