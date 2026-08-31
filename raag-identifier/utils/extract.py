"""Batch pitch-tracking of the dataset to a note/f0 cache.

Only the trackers touch audio. What we cache is deliberately *pre-tonic*: note events in
Hz plus the frame-level f0/voicing track, so that the tonic choice (clip / video / search)
stays a knob of the classifier rather than being baked into the cache.

    poetry run python -m utils.extract --tracker tony
    poetry run python -m utils.extract --tracker crepe

The tracker projects are located through `utils.config` (`RAAG_MELODY_DIR`,
`RAAG_SEPARATION_DIR`) and are resolved when a tracker actually runs -- importing this
module to *read* a cache requires neither of them, nor the audio.
"""

import argparse
import os
import time
from functools import lru_cache

import numpy as np
import librosa

from . import config
from .dataset import CLIP_RE, load_clips as _load_clips  # noqa: F401  (CLIP_RE re-exported)

#: Re-exported so the many callers that do `from utils.extract import DATA_VERSION` keep
#: working. The definitions live in `utils.config`, which is where dataset identity now
#: belongs; this module is about pitch tracking.
DATASET_ID = config.DATASET_ID
DATA_REVISIONS = config.DATA_REVISIONS
DATA_VERSION = config.DATA_VERSION
DATA_REVISION = config.DATA_REVISIONS.get(config.DATA_VERSION)


def data_dir():
    """The dataset directory. A function, not a constant, so a missing dataset raises
    where it is used rather than at import time -- importing this module to read a cache
    must not require the audio to be present."""
    return config.dataset_dir()


def cache_dir():
    return config.cache_dir()


def _add_tracker_paths():
    """Put melody-extraction (and source-separation) on sys.path. Called by the trackers,
    not at import time: reading a cache needs neither project installed."""
    melody = config.melody_dir()
    config.add_to_sys_path(melody, melody / "trackers" / "tony")


@lru_cache(maxsize=1)
def true_tonics():
    """{video: tonic_hz} from the dataset's hand annotation. Empty when unannotated."""
    return {c.video: c.tonic_hz for c in _load_clips(tonics_csv=config.tonics_csv())
            if c.video and c.tonic_hz is not None}


def list_clips():
    """Returns [{path, raag, split, video, chunk, clip_id, true_tonic_hz}, ...] by clip_id.

    Dicts rather than `utils.dataset.Clip` because a dozen call sites index them as dicts;
    the data comes from `utils.dataset.load_clips`, which reads `tonics.csv` rather than
    walking the tree.
    """
    d = config.dataset_dir()
    return [
        {
            "path": str(d / c.clip_id),
            "raag": c.raag,
            "split": c.split,
            "video": c.video,
            "chunk": c.chunk,
            "clip_id": c.clip_id,
            "true_tonic_hz": c.tonic_hz,
        }
        for c in _load_clips(audio_dir=d, tonics_csv=config.tonics_csv())
    ]


# ---------------------------------------------------------------- trackers


def _tony(audio, sr):
    """pYIN Vamp plugin: note HMM + smoothed frame track. Both kept, both in Hz."""
    _add_tracker_paths()
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

    _add_tracker_paths()
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


def cache_path(tracker, separate=None):
    suffix = "" if DATA_VERSION == "v0" else f"_{DATA_VERSION}"
    if separate and separate != "none":
        suffix += "_" + separate.replace("+", "-").replace(":", "-")
    return cache_dir() / f"notes_{tracker}{suffix}.npz"


def extract(tracker, limit=None, force=False, separate=None):
    """`separate` runs ../source-separation over each clip before tracking it, and lands in
    its own cache file so separated and unseparated runs never collide."""
    fn = TRACKERS[tracker]
    clips = list_clips()
    if limit:
        clips = clips[:limit]

    sep_fn = None
    if separate and separate != "none":
        config.add_to_sys_path(config.separation_dir())
        from separation import separate as _sep

        sep_fn = lambda a, sr: _sep(a, sr, backend=separate).melody

    out = {}
    dest = cache_path(tracker, separate)
    if dest.exists() and not force:
        with np.load(dest, allow_pickle=True) as z:
            out = {k: z[k] for k in z.files}
        print(f"resuming from {dest} ({len(out) // 4} clips already done)")

    t0 = time.time()
    done = 0
    for i, c in enumerate(clips):
        key = c["clip_id"]
        if f"{key}|notes" in out:
            continue
        try:
            audio, sr = librosa.load(c["path"], sr=None, mono=True)
            if sep_fn is not None:
                audio = sep_fn(audio, sr)
            notes, f0, voiced, hop = fn(audio, sr)
        except Exception as e:  # a handful of clips can be unreadable; skip loudly
            print(f"FAILED {key}: {type(e).__name__}: {e}")
            continue
        out[f"{key}|notes"] = notes
        out[f"{key}|f0"] = f0
        out[f"{key}|voiced"] = np.packbits(voiced)
        out[f"{key}|meta"] = np.array([hop, len(voiced)], dtype=np.float64)
        # stamp the cache with the dataset commit it was built from, so a stale cache can
        # be identified rather than silently trusted
        out["__revision__"] = np.array(str(DATA_REVISION))
        done += 1
        if done % 50 == 0:
            rate = (time.time() - t0) / done
            left = (len(clips) - i - 1) * rate
            print(f"  {i+1}/{len(clips)}  {rate:.2f}s/clip  eta {left/60:.1f}min", flush=True)
            np.savez_compressed(dest, **out)

    np.savez_compressed(dest, **out)
    print(f"wrote {dest}: {sum(1 for k in out if k.endswith('|notes'))} clips "
          f"from {DATA_VERSION} ({str(DATA_REVISION)[:10]}) in {(time.time()-t0)/60:.1f} min")


def load_cache(tracker, separate=None):
    """Returns {clip_id: {"notes": (N,3) float32 [t0,t1,hz], "f0": (T,), "voiced": (T,) bool, "hop": float}}."""
    with np.load(cache_path(tracker, separate), allow_pickle=True) as z:
        rev = str(z["__revision__"]) if "__revision__" in z.files else None
        if rev and DATA_REVISION and rev != str(DATA_REVISION):
            print(f"WARNING: {cache_path(tracker, separate).name} was built from dataset "
                  f"revision {rev[:10]}, but {DATA_VERSION} is {str(DATA_REVISION)[:10]}")
        keys = {k.split("|")[0] for k in z.files}
        out = {}
        for k in sorted(keys):
            if k.startswith("__") or f"{k}|notes" not in z.files:
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
    ap.add_argument("--separate", default=None,
                    help="source-separation backend to run first (see utils.config.separation_dir)")
    args = ap.parse_args()
    extract(args.tracker, limit=args.limit, force=args.force, separate=args.separate)
