"""Write down which *video* is in which split, so a second data source can reuse them exactly.

Every result in this survey was measured on the same partition of the dataset's 412 videos:

    test    the Hub's own test split -- 50 videos, 150 clips
    val     fold 0 of a round-robin deal of the train videos, per raag (`grouped_split`)
    fit     everything else in the Hub's train split

and the validation fold moves with the seed, which is how the seed replications re-dealt it.
Until now that partition only existed implicitly, as the output of a function over the Hub
clips. Training on the full recordings means reading from a directory where the videos are
plain files, and the only way to be sure the two sources agree is to write the partition
down once and have both read it back.

    poetry run python scripts/03_save_splits.py            # -> splits/v1.1_video_splits.csv
    poetry run python scripts/03_save_splits.py --check    # verify an existing file, write nothing

One row per video: raag, the Hub split, its fold under seeds 0-2, the annotated tonic, how
many Hub clips it has, and the full recording that matches it (by the 11-character video id
in the filename), with its duration. The full-audio directory is only *read*; `sf.info` on
each file is the whole interaction.

Videos in the full-audio directory that the Hub dataset does not contain are listed in a
second file rather than dropped silently -- they belong to no split, so nothing may train
on them without deciding where they go.
"""

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict

import soundfile as sf

import _bootstrap  # noqa: F401
from common.data import grouped_split, load_clips
from common.paths import FULL_AUDIO_DIR, SPLITS_FILE

SEEDS = (0, 1, 2)
VIDEO_RE = re.compile(r"\[([A-Za-z0-9_-]{11})\]\.mp3$")
COLUMNS = ["video", "raag", "label", "split", *(f"seed{s}" for s in SEEDS), "tonic_hz",
           "hf_clips", "audio", "seconds"]


def full_audio_index():
    """{video: [(raag folder, path), ...]} for every mp3 under FULL_AUDIO_DIR."""
    out = defaultdict(list)
    for p in sorted(FULL_AUDIO_DIR.glob("*/*.mp3")):
        m = VIDEO_RE.search(p.name)
        if m:
            out[m.group(1)].append((p.parent.name, p))
    return out


def build_rows():
    clips = load_clips()
    train = [c for c in clips if c.split == "train"]
    fold = {s: {c.video: "val" for c in grouped_split(train, val_frac=0.2, seed=s)[1]}
            for s in SEEDS}
    full = full_audio_index()

    by_video = defaultdict(list)
    for c in clips:
        by_video[c.video].append(c)

    rows, problems = [], []
    for video, cs in sorted(by_video.items()):
        c = cs[0]
        # a video filed under two raag folders is resolved by the Hub's label, and reported
        candidates = full.get(video, [])
        match = [p for raag, p in candidates if raag == c.raag]
        if len(candidates) > 1:
            problems.append(f"{video} is filed under {sorted(r for r, _ in candidates)}; "
                            f"the Hub labels it {c.raag}")
        if not match:
            problems.append(f"{video} ({c.raag}) has no full recording in {FULL_AUDIO_DIR}")
        path = match[0] if match else None
        rows.append({
            "video": video, "raag": c.raag, "label": c.label, "split": c.split,
            **{f"seed{s}": "test" if c.split == "test" else fold[s].get(video, "fit")
               for s in SEEDS},
            "tonic_hz": f"{c.tonic_hz:.4f}", "hf_clips": len(cs),
            "audio": str(path.relative_to(FULL_AUDIO_DIR)) if path else "",
            "seconds": f"{sf.info(str(path)).duration:.2f}" if path else "",
        })
    unassigned = [{"video": v, "raag_folder": raag, "audio": str(p.relative_to(FULL_AUDIO_DIR))}
                  for v, entries in sorted(full.items()) if v not in by_video
                  for raag, p in entries]
    return rows, unassigned, problems


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="compare with the file on disk")
    a = ap.parse_args()

    rows, unassigned, problems = build_rows()
    counts = Counter(r["seed0"] for r in rows)
    hours = sum(float(r["seconds"] or 0) for r in rows) / 3600
    print(f"{len(rows)} videos: {dict(counts)} at seed 0 | {hours:.1f} h of full audio | "
          f"{len(unassigned)} full recordings in no split")
    for p in problems:
        print(f"  NOTE {p}")

    if a.check:
        with open(SPLITS_FILE) as f:
            on_disk = list(csv.DictReader(f))
        same = [{k: str(v) for k, v in r.items()} for r in rows] == on_disk
        print(f"{SPLITS_FILE.name}: {'matches' if same else 'DIFFERS FROM'} a fresh derivation")
        sys.exit(0 if same else 1)

    SPLITS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SPLITS_FILE, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    extra = SPLITS_FILE.with_name(SPLITS_FILE.stem.replace("video_splits", "unassigned") + ".csv")
    with open(extra, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video", "raag_folder", "audio"])
        w.writeheader()
        w.writerows(unassigned)
    print(f"wrote {SPLITS_FILE} and {extra.name}")


if __name__ == "__main__":
    main()
