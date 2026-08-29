"""Batch transcription of hindustani-raag-small via ../melody-extraction, cached to disk.

Only the trackers in ../melody-extraction touch audio. What we cache is deliberately
*pre-tonic*: note events in Hz plus the frame-level f0/voicing track, so that the tonic
choice (clip / video / search) stays a knob of the classifier rather than being baked
into the cache.

    poetry run python extract.py --tracker tony
    poetry run python extract.py --tracker crepe
"""

import argparse
import csv
import os
import re
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import librosa

HERE = Path(__file__).resolve().parent
MELODY_DIR = HERE.parent / "melody-extraction"
CACHE_DIR = HERE / "cache"

# v1 (2026-08-28) changed the audio itself, not just the metadata: new chunk offsets,
# 20-60s instead of ~6s, 5/3 chunks per video instead of 3/2, 20 videos dropped and 6
# added, plus a hand-annotated `tonic_hz`. Nothing cached under v0 transfers, so the two
# versions get separate data dirs and separate caches and v0 stays reproducible.
DATA_VERSION = os.environ.get("RAAG_DATA_VERSION", "v1")
DATA_DIR = HERE.parent / ("hindustani-raag-small-v1" if DATA_VERSION == "v1"
                          else "hindustani-raag-small")

for p in (str(MELODY_DIR), str(MELODY_DIR / "trackers" / "tony")):
    if p not in sys.path:
        sys.path.insert(0, p)


CLIP_RE = re.compile(r"^(train|test)_\[(.+)\]_chunk(\d+)\.mp3$")


@lru_cache(maxsize=1)
def true_tonics():
    """{video: tonic_hz} from the dataset's hand annotation. Empty under v0."""
    f = DATA_DIR / "tonics.csv"
    if not f.exists():
        return {}
    with open(f) as fh:
        return {r["video"]: float(r["tonic_hz"]) for r in csv.DictReader(fh)}


def list_clips():
    """Returns [{path, raag, split, video, chunk, clip_id, true_tonic_hz}, ...] by clip_id."""
    tonics = true_tonics()
    clips = []
    for raag_dir in sorted(DATA_DIR.iterdir()):
        if not raag_dir.is_dir():
            continue
        for f in sorted(raag_dir.iterdir()):
            m = CLIP_RE.match(f.name)
            if not m:
                continue
            split, video, chunk = m.group(1), m.group(2), int(m.group(3))
            clips.append(
                {
                    "path": str(f),
                    "raag": raag_dir.name,
                    "split": split,
                    "video": video,
                    "chunk": chunk,
                    "clip_id": f"{raag_dir.name}/{f.name}",
                    "true_tonic_hz": tonics.get(video),
                }
            )
    return clips


# ---------------------------------------------------------------- trackers


def _tony(audio, sr):
    """pYIN Vamp plugin: note HMM + smoothed frame track. Both kept, both in Hz."""
    from trackers.tony.tony_tracker import _run_pyin, TONY_PARAMETERS

    f0_hz, voiced, hop, tony_notes = _run_pyin(audio, sr, dict(TONY_PARAMETERS))
    notes = np.array(
        [[n["t_start"], n["t_end"], n["f0_hz"]] for n in tony_notes if n["f0_hz"] > 0],
        dtype=np.float32,
    ).reshape(-1, 3)
    return notes, f0_hz.astype(np.float32), voiced, float(hop)


def _crepe(audio, sr):
    """torchcrepe frame track; notes come from melody-extraction's segment_notes.

    segment_notes works in cents, so we hand it a fake unit tonic (1 Hz) and convert the
    resulting note cents straight back to Hz — the real tonic is applied downstream.
    """
    import torch
    from trackers.crepe_tracker import (
        torchcrepe_predict,
        TARGET_SR,
        HOP_LENGTH,
        CONFIDENCE_THRESHOLD,
    )
    from note_segmentation import segment_notes

    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    wav = torch.from_numpy(np.ascontiguousarray(audio)).float().unsqueeze(0)
    with torch.no_grad():
        f0_hz, conf = torchcrepe_predict(wav, device=os.environ.get("CREPE_DEVICE", "cpu"))
    f0_hz = f0_hz.squeeze(0).numpy().astype(np.float32)
    conf = conf.squeeze(0).numpy()
    voiced = conf >= CONFIDENCE_THRESHOLD
    hop = HOP_LENGTH / TARGET_SR

    with np.errstate(divide="ignore", invalid="ignore"):
        cents = 1200.0 * np.log2(np.clip(f0_hz, 1e-6, None))
    segs = segment_notes(cents, voiced, hop, tol_cents=50.0, min_note_dur=0.2)
    notes = np.array(
        [[s.t_start, s.t_end, 2.0 ** (s.cents_relative / 1200.0)] for s in segs],
        dtype=np.float32,
    ).reshape(-1, 3)
    return notes, f0_hz, voiced, float(hop)


TRACKERS = {"tony": _tony, "crepe": _crepe}


# ---------------------------------------------------------------- driver


def cache_path(tracker):
    suffix = "" if DATA_VERSION == "v0" else f"_{DATA_VERSION}"
    return CACHE_DIR / f"notes_{tracker}{suffix}.npz"


def extract(tracker, limit=None, force=False):
    fn = TRACKERS[tracker]
    clips = list_clips()
    if limit:
        clips = clips[:limit]

    out = {}
    dest = cache_path(tracker)
    if dest.exists() and not force:
        with np.load(dest, allow_pickle=True) as z:
            out = {k: z[k] for k in z.files}
        print(f"resuming from {dest} ({len(out) // 4} clips already done)")

    CACHE_DIR.mkdir(exist_ok=True)
    t0 = time.time()
    done = 0
    for i, c in enumerate(clips):
        key = c["clip_id"]
        if f"{key}|notes" in out:
            continue
        try:
            audio, sr = librosa.load(c["path"], sr=None, mono=True)
            notes, f0, voiced, hop = fn(audio, sr)
        except Exception as e:  # a handful of clips can be unreadable; skip loudly
            print(f"FAILED {key}: {type(e).__name__}: {e}")
            continue
        out[f"{key}|notes"] = notes
        out[f"{key}|f0"] = f0
        out[f"{key}|voiced"] = np.packbits(voiced)
        out[f"{key}|meta"] = np.array([hop, len(voiced)], dtype=np.float64)
        done += 1
        if done % 50 == 0:
            rate = (time.time() - t0) / done
            left = (len(clips) - i - 1) * rate
            print(f"  {i+1}/{len(clips)}  {rate:.2f}s/clip  eta {left/60:.1f}min", flush=True)
            np.savez_compressed(dest, **out)

    np.savez_compressed(dest, **out)
    print(f"wrote {dest}: {len(out)//4} clips in {(time.time()-t0)/60:.1f} min")


def load_cache(tracker):
    """Returns {clip_id: {"notes": (N,3) float32 [t0,t1,hz], "f0": (T,), "voiced": (T,) bool, "hop": float}}."""
    with np.load(cache_path(tracker), allow_pickle=True) as z:
        keys = {k.split("|")[0] for k in z.files}
        out = {}
        for k in sorted(keys):
            if f"{k}|notes" not in z.files:
                continue
            hop, n_frames = z[f"{k}|meta"]
            out[k] = {
                "notes": z[f"{k}|notes"],
                "f0": z[f"{k}|f0"],
                "voiced": np.unpackbits(z[f"{k}|voiced"])[: int(n_frames)].astype(bool),
                "hop": float(hop),
            }
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracker", default="tony", choices=list(TRACKERS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    extract(args.tracker, limit=args.limit, force=args.force)
