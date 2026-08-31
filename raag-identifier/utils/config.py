"""Where the shared code looks for things it does not own.

Nothing in `utils/` derives a path by walking up from `__file__` and hoping a sibling
folder is there. Every external location is one named entry here, every entry is
overridable by an environment variable, and asking for one that is not on disk raises
with the variable you need to set.

    from utils import config

    config.dataset_dir()          # -> Path, or FileNotFoundError naming RAAG_DATASET_DIR
    config.cache_dir()            # -> Path, created if absent (it is ours to write)
    config.melody_dir()           # -> Path, or FileNotFoundError naming RAAG_MELODY_DIR

The defaults describe *this* checkout, because that is where these files came from and a
default that works beats a required variable for the common case. They are documented
guesses, not assumptions: nothing silently falls back to a wrong answer, and each accessor
tells you exactly which variable overrides it.

| what | variable | default |
|---|---|---|
| materialised dataset (audio + tonics.csv) | `RAAG_DATASET_DIR` | `<repo>/hindustani-raag-small-v1.1` |
| melody/note caches (we write here) | `RAAG_CACHE_DIR` | `<repo>/motif-classifier/cache` |
| pitch trackers | `RAAG_MELODY_DIR` | `<repo>/melody-extraction` |
| source separation | `RAAG_SEPARATION_DIR` | `<repo>/source-separation` |

`<repo>` is the directory holding `utils/`, i.e. `raag-identifier/`.
"""

import os
import sys
from pathlib import Path

#: The directory that contains `utils/`. Used only to build the *defaults* below.
REPO = Path(__file__).resolve().parent.parent

DATASET_ID = "neerajaabhyankar/hindustani-raag-small"

#: Version label -> the exact Hugging Face commit it means. The labels are ours and appear
#: throughout plan.md and results/; the SHAs are what actually make a run reproducible, so
#: fetching pins these rather than tracking `main`.
#:
#:   v0    2024-03-20  1253 clips of ~6 s, no tonic column
#:   v1    2026-08-28  new audio: 1960 clips of 20-60 s, real train/test splits, tonic_hz
#:   v1.1  2026-08-31  v1 with six tonic annotations corrected; audio byte-identical to v1
DATA_REVISIONS = {
    "v0": "0dfb021e54e0e7489b90a47e23ef15f34fa740ec",
    "v1": "9944c647cb733573fcc5bb05297e1622fc1867f2",
    "v1.1": "326caef0bc01da44ad46e4d9c65a5146da6bcc5b",
}

DATA_VERSION = os.environ.get("RAAG_DATA_VERSION", "v1.1")


def revision(version=None):
    """The pinned commit SHA for a version label. Raises on an unknown label."""
    v = version or DATA_VERSION
    try:
        return DATA_REVISIONS[v]
    except KeyError:
        raise KeyError(f"unknown data version {v!r}; known: {sorted(DATA_REVISIONS)}") from None


def _default_dataset_dir():
    v = DATA_VERSION
    return REPO / ("hindustani-raag-small" if v == "v0" else f"hindustani-raag-small-{v}")


def _resolve(env_var, default, what, must_exist=True, create=False, hint=None):
    p = Path(os.environ[env_var]).expanduser() if os.environ.get(env_var) else default
    if create:
        p.mkdir(parents=True, exist_ok=True)
    if must_exist and not p.exists():
        msg = [f"{what} not found at {p}", f"    set {env_var} to point at it"]
        if hint:
            msg.append(f"    {hint}")
        raise FileNotFoundError("\n".join(msg))
    return p


def dataset_dir(must_exist=True):
    """The materialised dataset: `<Raag>/*.mp3` plus `tonics.csv` and `REVISION`."""
    return _resolve(
        "RAAG_DATASET_DIR", _default_dataset_dir(),
        f"dataset {DATA_VERSION}", must_exist,
        hint="materialise it with: python -m utils.dataset fetch --version "
             f"{DATA_VERSION} --audio-dir <path>",
    )


def tonics_csv(must_exist=True):
    """`tonics.csv` inside the dataset dir -- clip_id, raag, split, video, chunk, tonic_hz."""
    p = dataset_dir(must_exist=must_exist) / "tonics.csv"
    if must_exist and not p.exists():
        raise FileNotFoundError(
            f"{p} is missing -- the dataset dir exists but was not written by "
            f"utils.dataset.fetch(). Re-fetch, or set RAAG_DATASET_DIR to a complete one."
        )
    return p


def cache_dir():
    """Where note/f0 extraction caches live. Ours to write, so it is created on demand."""
    return _resolve("RAAG_CACHE_DIR", REPO / "motif-classifier" / "cache",
                    "cache directory", must_exist=False, create=True)


def melody_dir():
    """The pitch-tracker project. Only needed to *extract*; not to read a cache."""
    return _resolve("RAAG_MELODY_DIR", REPO / "melody-extraction", "melody-extraction")


def separation_dir():
    """The source-separation project. Only needed when a caller asks for a stem."""
    return _resolve("RAAG_SEPARATION_DIR", REPO / "source-separation", "source-separation")


def add_to_sys_path(*paths):
    """Idempotently put directories on `sys.path`. Explicit, and never at import time."""
    for p in paths:
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
