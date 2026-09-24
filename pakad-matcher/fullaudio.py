"""The full recordings the dataset chunks were cut from -- read-only, for realistic context.

`../raag-identifier/hindustani-raag-fullaudios/<Raag>/... [<video>].mp3`. The video id in the
filename is the same one `tonics.csv` annotates, so every full audio inherits a **hand-annotated
tonic**. Test-split videos are excluded here, so annotation never touches them.

    poetry run python fullaudio.py            # f0 cache for config.FOCUS_RAAGS' train videos
"""

import re
from dataclasses import dataclass
from functools import lru_cache
from multiprocessing import Pool

import numpy as np

import _bootstrap  # noqa: F401
import config as C
import contour as clip_contour
from utils import config as ucfg

VIDEO_RE = re.compile(r"\[([^\]]+)\]\.mp3$")


@dataclass(frozen=True)
class FullAudio:
    raag: str
    video: str
    path: str
    tonic_hz: float


@lru_cache(maxsize=1)
def _video_meta():
    """{video: (raag, split, tonic_hz)} from the pinned dataset's tonics.csv."""
    from utils import dataset
    out = {}
    for c in dataset.load_clips(tonics_csv=ucfg.tonics_csv()):
        if c.video:
            out[c.video] = (c.raag, c.split, c.tonic_hz)
    return out


@lru_cache(maxsize=1)
def index(raags=None):
    """Train-split full audios we can use, by video id."""
    meta, out = _video_meta(), {}
    for d in sorted(C.FULLAUDIO_DIR.iterdir()):
        if not d.is_dir() or (raags and d.name not in raags):
            continue
        for f in sorted(d.glob("*.mp3")):
            m = VIDEO_RE.search(f.name)
            if not m or m.group(1) not in meta:
                continue
            raag, split, tonic = meta[m.group(1)]
            if split != "train" or tonic is None:
                continue
            out[m.group(1)] = FullAudio(raag, m.group(1), str(f), tonic)
    return out


def _melodia(audio, sr):
    """Melodia with `utils.extract._essentia`'s settings, but keeping **pitch salience**.

    That shared helper drops the confidence array; the annotations say we need it -- most of
    the "no"s are the tanpura or a stray accompaniment note that Melodia tracked while the
    voice was quiet, and salience is what tells those apart.
    """
    import essentia
    import essentia.standard as es
    import librosa

    essentia.log.warningActive = False
    target_sr, hop_samples = 44100, 196
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    f0, conf = es.PredominantPitchMelodia(
        sampleRate=target_sr, frameSize=2048, hopSize=hop_samples, binResolution=10,
        guessUnvoiced=False)(es.EqualLoudness(sampleRate=target_sr)(audio))
    return (np.asarray(f0, np.float32), np.asarray(conf, np.float32),
            hop_samples / target_sr)


def _track(video):
    """Melodia over a long file, in blocks, so memory stays flat."""
    import librosa

    fa = index()[video]
    dur = librosa.get_duration(path=fa.path)
    f0, conf, hop = [], [], None
    for start in np.arange(0.0, dur, C.FULLAUDIO_BLOCK_S):
        audio, sr = librosa.load(fa.path, sr=None, mono=True, offset=float(start),
                                 duration=C.FULLAUDIO_BLOCK_S)
        if len(audio) < sr:
            break
        block, c, hop = _melodia(audio, sr)
        f0.append(block); conf.append(c)
    return video, np.concatenate(f0), np.concatenate(conf), hop


def extract(raags=None):
    C.CACHE_DIR.mkdir(exist_ok=True)
    done = {}
    if C.FULLAUDIO_CACHE.exists():
        with np.load(C.FULLAUDIO_CACHE) as z:
            done = {k: z[k] for k in z.files}
    todo = [v for v in index(raags) if f"{v}|conf" not in done]
    print(f"{len(done) // 2} cached, {len(todo)} to go")
    with Pool(C.EXTRACT_WORKERS) as pool:
        for i, (v, f0, conf, hop) in enumerate(pool.imap_unordered(_track, todo), 1):
            done[f"{v}|f0"], done[f"{v}|hop"] = f0, np.float64(hop)
            done[f"{v}|conf"] = conf
            np.savez_compressed(C.FULLAUDIO_CACHE, **done)
            print(f"  {i}/{len(todo)}  {v}  {len(f0) * hop / 60:.1f} min", flush=True)


@lru_cache(maxsize=1)
def _cache():
    with np.load(C.FULLAUDIO_CACHE) as z:
        return {k: z[k] for k in z.files}


def contour(video, downsample=C.DOWNSAMPLE):
    """Same Contour type the clip path uses, so the matcher does not care which it gets."""
    z, fa = _cache(), index()[video]
    return clip_contour.contour_from_f0(z[f"{video}|f0"], float(z[f"{video}|hop"]),
                                        fa.tonic_hz, video, downsample)


def salience(video, downsample=C.DOWNSAMPLE):
    """Melodia's per-frame pitch salience, downsampled to match `contour`."""
    c = _cache()[f"{video}|conf"]
    n = len(c) // downsample * downsample
    return c[:n].reshape(-1, downsample).mean(axis=1)


def cached_videos(raags=None):
    have = {k[:-5] for k in _cache() if k.endswith("|conf")}
    return sorted(v for v in index(raags) if v in have)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--raags", nargs="+", default=C.ANNOTATION_RAAGS)
    a = ap.parse_args()
    idx = index(tuple(a.raags))
    print(f"{len(idx)} train-split full audios over {len({f.raag for f in idx.values()})} raags")
    extract(tuple(a.raags))
