"""The clip index, and splits that are always grouped by video.

The one rule this module exists to enforce: **a video never straddles a split boundary.**
The dataset's clips come in ~3-chunk groups cut from a single recording, so a clip whose
siblings are in the training fold is not a test of raag recognition -- it is a test of
recording recall, and it inflates accuracy badly. Both earlier DL attempts in this repo
used `train_test_split(stratify_by_column="label")`, which shuffles at the clip level; their
validation numbers should be read with that in mind.

There is deliberately no function here that returns a clip-level shuffle.

    from common import data
    clips = data.load_clips("train")
    tr, va = data.grouped_split(clips, val_frac=0.2, seed=0)
    folds  = data.grouped_folds(clips, n_folds=5, seed=0)
"""

import csv
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .paths import DATA_DIR, check_data


@dataclass(frozen=True)
class Clip:
    """One 20-second chunk, with everything known about it before any audio is read."""

    clip_id: str      # "Bageshree/train_[abc123]_chunk0.mp3" -- also the cache key
    raag: str         # folder name, e.g. "Bageshree"
    label: int        # index into LABELS
    split: str        # "train" | "test"
    video: str        # youtube id; the grouping key for every split in this project
    chunk: int
    tonic_hz: float   # hand-annotated in v1.1, constant across a video's chunks

    @property
    def path(self) -> Path:
        return DATA_DIR / self.clip_id


@lru_cache(maxsize=1)
def _index():
    """(LABELS, all clips) read once from tonics.csv."""
    check_data()
    with open(DATA_DIR / "tonics.csv") as fh:
        rows = list(csv.DictReader(fh))
    labels = sorted({r["raag"] for r in rows})
    label_of = {r: i for i, r in enumerate(labels)}
    clips = [
        Clip(
            clip_id=r["clip_id"],
            raag=r["raag"],
            label=label_of[r["raag"]],
            split=r["split"],
            video=r["video"],
            chunk=int(r["chunk"]),
            tonic_hz=float(r["tonic_hz"]),
        )
        for r in rows
    ]
    return labels, sorted(clips, key=lambda c: c.clip_id)


def labels():
    """The 50 raag folder names, sorted. Index into this is the class id everywhere."""
    return list(_index()[0])


def load_clips(split=None):
    """All clips, or just one split. `split="test"` is the held-out 150 -- see RUNS.md."""
    _, clips = _index()
    if split is None:
        return list(clips)
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train', 'test' or None, got {split!r}")
    return [c for c in clips if c.split == split]


def grouped_folds(clips, n_folds=5, seed=0):
    """{video: fold}. Videos are dealt round-robin *within each raag*, so every fold sees
    every raag. Same construction as ../motif-classifier/evaluate.py, so the fold
    assignments are comparable between the two projects at the same seed."""
    by_raag = defaultdict(list)
    for c in clips:
        by_raag[c.raag].append(c.video)
    rng = np.random.default_rng(seed)
    fold_of_video = {}
    for raag, videos in sorted(by_raag.items()):
        vids = sorted(set(videos))
        rng.shuffle(vids)
        for i, v in enumerate(vids):
            fold_of_video[v] = i % n_folds
    return fold_of_video


def fold_indices(clips, n_folds=5, seed=0):
    """Yields (fold, train_clips, val_clips) for each fold -- the CV harness."""
    fov = grouped_folds(clips, n_folds, seed)
    for k in range(n_folds):
        tr = [c for c in clips if fov[c.video] != k]
        va = [c for c in clips if fov[c.video] == k]
        yield k, tr, va


def grouped_split(clips, val_frac=0.2, seed=0):
    """A single train/val split, grouped by video and stratified by raag.

    Implemented as one slice of an n-fold deal so that `grouped_split(val_frac=0.2)` and
    fold 0 of `fold_indices(n_folds=5)` are the *same split* -- one less thing to get
    subtly inconsistent between a quick run and a CV run.
    """
    n_folds = max(2, int(round(1.0 / val_frac)))
    fov = grouped_folds(clips, n_folds, seed)
    tr = [c for c in clips if fov[c.video] != 0]
    va = [c for c in clips if fov[c.video] == 0]
    return tr, va


def summarise(clips):
    """One line describing a clip set -- printed at the top of every run."""
    return (
        f"{len(clips)} clips, {len({c.raag for c in clips})} raags, "
        f"{len({c.video for c in clips})} videos"
    )


if __name__ == "__main__":
    L = labels()
    allc = load_clips()
    print(f"{len(L)} labels; all: {summarise(allc)}")
    for s in ("train", "test"):
        print(f"  {s:5s} {summarise(load_clips(s))}")
    tr, va = grouped_split(load_clips("train"))
    print(f"  grouped split -> train {summarise(tr)} | val {summarise(va)}")
    overlap = {c.video for c in tr} & {c.video for c in va}
    print(f"  video overlap between train and val: {len(overlap)} (must be 0)")
