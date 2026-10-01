"""S0: frame-level f0 cache, and the tonic-relative cents contour the matcher works on.

The cache holds raw f0 (Hz) only -- no note segmentation (plan.md: the shared segmenter's
merge destroys swar identity) and no tonic (applied on load, from tonics.csv).

    poetry run python contour.py            # extract the train split (~10 min)
"""

from dataclasses import dataclass
from functools import lru_cache
from multiprocessing import Pool

import numpy as np

import _bootstrap  # noqa: F401
import config as C
from utils import config as ucfg, dataset


@lru_cache(maxsize=1)
def clips():
    """{clip_id: Clip} for the configured split."""
    cs = dataset.load_clips(audio_dir=ucfg.dataset_dir(), tonics_csv=ucfg.tonics_csv())
    return {c.clip_id: c for c in cs if c.split == C.SPLIT}


def _track(clip_id):
    import librosa
    from utils.extract import _essentia

    audio, sr = librosa.load(clips()[clip_id].path, sr=None, mono=True)
    _notes, f0, _voiced, hop = _essentia(audio, sr)
    return clip_id, f0.astype(np.float32), hop


def extract():
    done = {}
    if C.F0_CACHE.exists():
        with np.load(C.F0_CACHE) as z:
            done = {k: z[k] for k in z.files}
    todo = [c for c in clips() if f"{c}|f0" not in done]
    print(f"{len(done) // 2} cached, {len(todo)} to go")
    with Pool(C.EXTRACT_WORKERS) as pool:
        for i, (cid, f0, hop) in enumerate(pool.imap_unordered(_track, todo), 1):
            done[f"{cid}|f0"], done[f"{cid}|hop"] = f0, np.float64(hop)
            if i % 100 == 0 or i == len(todo):
                np.savez_compressed(C.F0_CACHE, **done)
                print(f"  {i}/{len(todo)}", flush=True)


@lru_cache(maxsize=1)
def _cache():
    with np.load(C.F0_CACHE) as z:
        return {k: z[k] for k in z.files}


@dataclass
class Contour:
    clip_id: str
    cents: np.ndarray    # pitch re Sa, cents; NaN where unvoiced
    hop: float           # seconds per frame (after downsampling)

    @property
    def times(self):
        return np.arange(len(self.cents)) * self.hop


def contour(clip_id, downsample=C.DOWNSAMPLE):
    """Tonic-relative cents, downsampled by a NaN-aware median (unvoiced if most frames are)."""
    z = _cache()
    return contour_from_f0(z[f"{clip_id}|f0"], float(z[f"{clip_id}|hop"]),
                           clips()[clip_id].tonic_hz, clip_id, downsample)


def contour_from_f0(f0, hop, tonic, ident, downsample=C.DOWNSAMPLE):
    with np.errstate(divide="ignore", invalid="ignore"):
        cents = np.where(f0 > 0, 1200 * np.log2(f0 / tonic), np.nan)
    n = len(cents) // downsample * downsample
    blocks = cents[:n].reshape(-1, downsample)
    voiced = np.sum(~np.isnan(blocks), axis=1) > downsample // 2
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(blocks, axis=1)
    return Contour(ident, np.where(voiced, med, np.nan), hop * downsample)


def cached_ids():
    return sorted(k[:-3] for k in _cache() if k.endswith("|f0"))


if __name__ == "__main__":
    extract()
