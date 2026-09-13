"""Cache the Sa-anchored CQT of every full recording in the splits file.

    poetry run python scripts/04_build_fullaudio_cache.py              # all 412, 3 processes
    poetry run python scripts/04_build_fullaudio_cache.py --roles fit  # only what training reads
    poetry run python scripts/04_build_fullaudio_cache.py --workers 5

About 25 minutes of single-core work for the 159 h, 1.8 GB on disk (uint8; see
`common/fullaudio.py` for why that loses nothing). Resumable: a recording already cached is
skipped, and each file is written atomically, so a killed build never leaves a truncated
array behind for the next run to trust.

Reads `paths.FULL_AUDIO_DIR`; never writes to it.
"""

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import _bootstrap  # noqa: F401
from common import fullaudio as F


def _build(video_id):
    v = F.by_id()[video_id]
    t0 = time.time()
    F.build_cqt(v)
    return video_id, v.seconds, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roles", nargs="*", default=["fit", "val", "test"],
                    help="which seed-0 roles to cache")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()

    todo = [v for v in F.load_videos() if v.role(0) in a.roles and not F.cqt_path(v).exists()]
    hours = sum(v.seconds for v in todo) / 3600
    print(f"{len(todo)} recordings to cache ({hours:.1f} h of audio), {a.workers} processes "
          f"-> {F.CQT_DIR}", flush=True)
    t0, done_s = time.time(), 0.0
    with ProcessPoolExecutor(a.workers) as pool:
        futures = [pool.submit(_build, v.video) for v in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            vid, secs, took = fut.result()
            done_s += secs
            if i % 20 == 0 or i == len(todo):
                rate = done_s / max(time.time() - t0, 1e-9)
                eta = (hours * 3600 - done_s) / max(rate, 1e-9) / 60
                print(f"  {i}/{len(todo)}  {done_s / 3600:.1f} h done  eta {eta:.0f} min",
                      flush=True)
    size = sum(p.stat().st_size for p in F.CQT_DIR.glob("*.npy")) / 1e9
    print(f"done in {(time.time() - t0) / 60:.0f} min; cache holds {size:.2f} GB")


if __name__ == "__main__":
    main()
