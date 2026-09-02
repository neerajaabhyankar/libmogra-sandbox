"""Where everything is, in one place.

Shared code lives in `raag-identifier/utils/` and is imported as `utils.*` -- the graded
evaluation in particular is the thing this project is judged by, and a second copy of it
would drift. `add_sibling_paths()` remains only for the two projects that are *not* shared
utilities: motif-classifier (for `methods.m12_dbhist`, the Stage 4 DB prior) and
source-separation (Stage 3).
"""

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent          # survey-aug-2026/
REPO = HERE.parent                                      # raag-identifier/

#: The pinned dataset revision this whole folder is about.
DATA_VERSION = "v1.1"
DATA_REVISION = "326caef0bc01da44ad46e4d9c65a5146da6bcc5b"
DATASET_ID = "neerajaabhyankar/hindustani-raag-small"

#: Materialised by `utils.dataset.fetch`; override with RAAG_DATASET_DIR.
DATA_DIR = Path(os.environ["RAAG_DATASET_DIR"]).expanduser() \
    if os.environ.get("RAAG_DATASET_DIR") else REPO / f"hindustani-raag-small-{DATA_VERSION}"

CACHE = HERE / "cache"
RESULTS = HERE / "results" / DATA_VERSION

# sibling projects we import from (read-only -- nothing here writes to them)
MOTIF_DIR = REPO / "motif-classifier"
SEP_DIR = REPO / "source-separation"
MELODY_DIR = REPO / "melody-extraction"
JEEVSTER_DIR = REPO / "carnatic-raga-classifier-jeevster"
RESNET_DIR = REPO / "hindustani-raag-classifier-resnet"


def add_sibling_paths():
    """Idempotently put the sibling projects on sys.path. Call before importing from them.

    `utils` is not in this list -- it is reached as a normal package via `UTILS_ROOT`
    below, so a missing shared module is an ImportError at the top of the file rather
    than a surprise at runtime.
    """
    for p in (MOTIF_DIR, SEP_DIR, MELODY_DIR, RESNET_DIR):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


#: `raag-identifier/` on the path, so `import utils` resolves however this was launched.
UTILS_ROOT = REPO
if str(UTILS_ROOT) not in sys.path:
    sys.path.insert(0, str(UTILS_ROOT))


def check_data():
    """Fail loudly and usefully if the pinned revision is not on disk."""
    rev_file = DATA_DIR / "REVISION"
    if not rev_file.exists():
        raise FileNotFoundError(
            f"{DATA_DIR} is missing. Materialise it first:\n"
            f"    python -m utils.dataset fetch --version {DATA_VERSION} "
            f"--audio-dir {DATA_DIR}\n"
            f"or from python:\n"
            f"    from utils import dataset\n"
            f"    dataset.fetch(repo_id={DATASET_ID!r}, revision={DATA_REVISION!r},\n"
            f"                  audio_dir={str(DATA_DIR)!r},\n"
            f"                  tonics_csv={str(DATA_DIR / 'tonics.csv')!r},\n"
            f"                  revision_file={str(DATA_DIR / 'REVISION')!r})"
        )
    got = rev_file.read_text().strip()
    want = f"{DATASET_ID}@{DATA_REVISION}"
    if got != want:
        raise RuntimeError(f"{DATA_DIR} holds {got!r}, expected {want!r}")
    return DATA_DIR
