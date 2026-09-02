"""Training data: one pinned Hugging Face revision, cached as features.

Only `train.py` needs this. Inference never touches it.

The revision is pinned to a commit rather than a branch, and that is not pedantry for this
dataset: the v1 audio lives **only** in the parquet files, while the `<Raag>/*.mp3` tree at
the repo root is still v0. A loader that walks the raw layout silently trains on the wrong
corpus. `datasets.load_dataset` reads the parquet, so this cannot happen here.

Both branches' inputs are cached per clip, because 34 epochs over 1810 clips would
otherwise decode and transform the same audio 34 times. About 250 MB for the full corpus.
"""

import io
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

DATASET_ID = "neerajaabhyankar/hindustani-raag-small"
REVISION = "326caef0bc01da44ad46e4d9c65a5146da6bcc5b"     # v1.1
CLIP_RE = re.compile(r"^(train|test)_\[(.+)\]_chunk(\d+)\.mp3$")


@dataclass
class Clip:
    clip_id: str        # "<Raag>/<filename>"
    raag: str
    split: str          # train | test
    video: str          # source recording -- splits are grouped by this, never by clip
    tonic_hz: float


def stream(repo_id=DATASET_ID, revision=REVISION):
    """Yield `(raag, filename, mp3_bytes, tonic_hz)` from the Hub, audio undecoded."""
    from datasets import Audio, load_dataset

    ds = load_dataset(repo_id, revision=revision)
    for key in sorted(ds.keys()) if hasattr(ds, "keys") else [None]:
        part = ds[key] if key is not None else ds
        names = getattr(part.features.get("label"), "names", None)
        part = part.cast_column("audio", Audio(decode=False))
        for row in part:
            label = row["label"]
            yield (names[int(label)] if names else str(label),
                   Path(row["audio"]["path"]).name, row["audio"]["bytes"],
                   float(row["tonic_hz"]))


def build_cache(cache_dir, limit=None, device="cpu", progress=print):
    """Materialise both branches' inputs for every clip. Resumable: existing files are kept.

    `limit` caps how many clips are taken per raag per split, for smoke tests.

    Returns the clip index, which is also written to `<cache_dir>/index.json`.
    """
    import librosa

    from . import audio as A
    from . import cqt_branch, melody_branch

    cache_dir = Path(cache_dir)
    (cache_dir / "cqt").mkdir(parents=True, exist_ok=True)
    (cache_dir / "melody").mkdir(parents=True, exist_ok=True)

    clips, done, seen = [], 0, {}
    for raag, filename, blob, tonic_hz in stream():
        m = CLIP_RE.match(filename)
        clip = Clip(f"{raag}/{filename}", raag, m.group(1), m.group(2), tonic_hz)
        key = (clip.raag, clip.split)
        seen[key] = seen.get(key, 0) + 1
        if limit and seen[key] > limit:     # per raag *and* split, so a smoke test
            continue                        # still sees all 50 classes
        clips.append(clip)

        cqt_path, mel_path = _paths(cache_dir, clip)
        if cqt_path.exists() and mel_path.exists():
            continue
        y, sr = librosa.load(io.BytesIO(blob), sr=None, mono=True)
        if not cqt_path.exists():
            y22 = A.fit_length(A.peak_normalise(A.resample(y, sr, A.SR_CQT)),
                               int(round(A.SR_CQT * A.WINDOW_SECONDS)))
            np.save(cqt_path, cqt_branch.features(y22, tonic_hz)[0].astype(np.float16))
        if not mel_path.exists():
            f0, voiced = melody_branch.f0_track(A.resample(y, sr, melody_branch.SR),
                                                device=device)
            np.save(mel_path, melody_branch.histogram(f0, voiced, tonic_hz).astype(np.float32))
        done += 1
        if done % 50 == 0:
            progress(f"  cached {done} clips ({len(clips)} seen)")

    (cache_dir / "index.json").write_text(json.dumps([asdict(c) for c in clips], indent=1))
    progress(f"cache ready: {len(clips)} clips in {cache_dir}")
    return clips


def _paths(cache_dir, clip):
    stem = clip.clip_id.replace("/", "__").replace(".mp3", "")
    return Path(cache_dir) / "cqt" / f"{stem}.npy", Path(cache_dir) / "melody" / f"{stem}.npy"


def load_index(cache_dir):
    return [Clip(**d) for d in json.loads((Path(cache_dir) / "index.json").read_text())]


def load_features(cache_dir, clips):
    """(n, 1, 144, 431) float32 CQT windows and (n, 120) float32 histograms."""
    pairs = [_paths(cache_dir, c) for c in clips]
    cqt = np.stack([np.load(p).astype(np.float32) for p, _ in pairs])[:, None]
    mel = np.stack([np.load(p) for _, p in pairs])
    return cqt, mel


def grouped_split(clips, val_fraction=0.2, seed=0):
    """Split by *video*, stratified by raag. Never by clip.

    Three chunks come from each recording. A split that puts one chunk in train and its
    siblings in validation measures recording recall, not raag recognition -- it inflated
    an early version of this work by about 20 points before it was caught.

    Videos are dealt round-robin within each raag and fold 0 becomes the validation set,
    which is exactly how the reported validation numbers were produced.
    """
    rng = np.random.default_rng(seed)
    n_folds = max(2, int(round(1.0 / val_fraction)))
    by_raag = {}
    for c in clips:
        by_raag.setdefault(c.raag, set()).add(c.video)
    fold = {}
    for raag, videos in sorted(by_raag.items()):
        videos = sorted(videos)
        rng.shuffle(videos)
        for i, v in enumerate(videos):
            fold[v] = i % n_folds
    return ([c for c in clips if fold[c.video] != 0],
            [c for c in clips if fold[c.video] == 0])
