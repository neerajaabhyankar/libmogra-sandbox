"""Build the audio and CQT caches. Run this once before anything else.

Decoding an mp3 costs ~200 ms and a CQT ~600 ms; at 1960 clips that is 26 minutes per epoch
wasted if it happens inside the training loop. This pays it once.

What gets written, all under `cache/`:

    audio/raw/<clip>.npy      22.05 kHz mono int16, peak-normalised   ~1.7 GB
    audio/hpss/<clip>.npy     the same after HPSS melody separation   ~1.7 GB  (--separate)
    cqt/anchor_36x4/<clip>.npy   Sa-anchored log-CQT, float16         ~250 MB
    cqt/none_36x4/<clip>.npy     fixed-fmin log-CQT (the control)     ~250 MB

Everything is resumable: a clip whose .npy already exists is skipped, so a killed run
continues where it stopped. That is also what makes this safe to call from a Colab notebook
that has crashed once already.

    poetry run python scripts/00_build_cache.py --dry-run
    poetry run python scripts/00_build_cache.py                      # audio + both CQTs
    poetry run python scripts/00_build_cache.py --separate hpss      # add the HPSS variants
"""

import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import _bootstrap  # noqa: F401
from common import audio
from common.data import load_clips, summarise
from common.paths import CACHE


def _one(args):
    """(clip, what, separate) -> bytes written. Top-level so it is picklable."""
    clip, what, separate = args
    try:
        if what == "audio":
            audio.cached_waveform(clip, separate=separate)
        elif what == "cqt_anchor":
            audio.cached_cqt(clip, tonic="anchor", separate=separate)
        elif what == "cqt_none":
            audio.cached_cqt(clip, tonic="none", separate=separate)
        return None
    except Exception as e:  # one bad mp3 must not kill a 30-minute build
        return f"{clip.clip_id} [{what}]: {type(e).__name__}: {e}"


def build(clips, what, separate, workers, dry_run):
    todo = [(c, what, separate) for c in clips]
    label = f"{what}/{separate or 'raw'}"
    if dry_run:
        print(f"  {label:20s} {len(todo)} clips")
        return []
    t0, errors, done = time.time(), [], 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for err in ex.map(_one, todo, chunksize=8):
            done += 1
            if err:
                errors.append(err)
            if done % 200 == 0 or done == len(todo):
                rate = done / max(time.time() - t0, 1e-9)
                print(f"  {label:20s} {done}/{len(todo)}  {rate:.1f} clips/s  "
                      f"eta {(len(todo) - done) / max(rate, 1e-9) / 60:.1f} min", flush=True)
    for e in errors[:10]:
        print(f"    ERROR {e}")
    if len(errors) > 10:
        print(f"    ... and {len(errors) - 10} more")
    return errors


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--separate", default=None,
                    help="also build a separated variant: hpss | hpss+drone | demucs")
    ap.add_argument("--skip-cqt", action="store_true", help="waveform cache only")
    ap.add_argument("--workers", type=int, default=6, help="processes (M1 has 8 cores)")
    ap.add_argument("--limit", type=int, default=None, help="first N clips, for a smoke test")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    clips = load_clips()
    if args.limit:
        clips = clips[:args.limit]
    print(f"cache -> {CACHE}")
    print(f"clips: {summarise(clips)}")

    jobs = [("audio", None)]
    if not args.skip_cqt:
        jobs += [("cqt_anchor", None), ("cqt_none", None)]
    if args.separate:
        jobs.append(("audio", args.separate))
        if not args.skip_cqt:
            jobs.append(("cqt_anchor", args.separate))

    errors = []
    for what, sep in jobs:
        errors += build(clips, what, sep, args.workers, args.dry_run)

    if not args.dry_run:
        total = sum(f.stat().st_size for f in CACHE.rglob("*.npy"))
        print(f"\ncache size: {total / 1e9:.2f} GB, {len(errors)} errors")


if __name__ == "__main__":
    main()
