"""Training from the full recordings instead of the Hub's 20 s clips.

The Hub dataset keeps about 100 s of each recording -- five clips per training video, three
per test video -- out of a median of 20 minutes. The recordings themselves are on disk
(`paths.FULL_AUDIO_DIR`: 412 of them, 159 h), so how much this corpus can teach is partly a
question of how much of it the model is ever shown.

    videos(role, seed)      the splits file read back -- the only source of truth for
                            which video is fit / val / test, shared with the Hub clips
    build_cqt(video)        the whole recording's Sa-anchored CQT, computed once and cached
                            as uint8 (1.8 GB for all 412)
    FullAudioCQTDataset     `windows_per_video` 20 s windows per video per epoch, drawn
                            fresh each time an item is read, so no two epochs see the same
                            crops; or a fixed, evenly spaced set -- for evaluation, and for
                            training as "a dataset with N clips per performance"

**The features are the Hub path's features.** A window is cut from the cached whole-recording
CQT and normalised exactly as `audio.cqt` normalises a clip -- decibels relative to *that
window's* loudest bin, floored 80 dB down, mapped to [0, 1] -- so a model trained here reads
the same input as one trained on clips, and both are selected and scored on the same Hub
val/test clips. The only difference is storage: 0.47 dB steps (uint8 over 120 dB), below
anything the network could be using.

**What is filtered, and why only gently.** A concert video opens with tuning, announcements
and applause, and has silences inside it. Windows start only at positions (a) at least
`trim_seconds` from either end and (b) mostly loud -- at least `loud_fraction` of their
frames within `loudness_db` of the recording's median frame level. That removes dead air
and the video's edges. It deliberately does not try to detect speech or applause, which
would need a classifier and a threshold nobody has checked.

The recordings directory is only ever read. The cache is written under `paths.CACHE`.
"""

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from . import audio
from . import tonic as tonic_mod
from .data import Clip
from .datasets import _Base
from .paths import CACHE, FULL_AUDIO_DIR, SPLITS_FILE

CQT_DIR = CACHE / "fullcqt"
DB_RANGE = 120.0                                  # stored decibels below the recording's peak
WINDOW_FRAMES = 431                               # 20 s: what every Hub-trained model saw
FRAMES_PER_SECOND = audio.SR_CQT / audio.CQT_HOP  # 21.5
N_BINS = audio.CQT_BINS_PER_OCTAVE * audio.CQT_OCTAVES


# ------------------------------------------------------------------ the splits, read back


@dataclass(frozen=True)
class Video:
    video: str
    raag: str
    label: int
    split: str          # the Hub's: train | test
    roles: tuple        # fit | val | test, under seed 0, 1, 2
    tonic_hz: float
    audio: Path
    seconds: float

    def role(self, seed=0):
        if seed >= len(self.roles):
            raise ValueError(f"the splits file records seeds 0-{len(self.roles) - 1}; "
                             f"re-run scripts/03_save_splits.py with seed {seed} added")
        return self.roles[seed]


@lru_cache(maxsize=1)
def load_videos(path=SPLITS_FILE):
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} missing; write it with scripts/03_save_splits.py")
    with open(path) as f:
        rows = list(csv.DictReader(f))
    seeds = sorted(int(k[4:]) for k in rows[0] if k.startswith("seed"))
    return tuple(Video(r["video"], r["raag"], int(r["label"]), r["split"],
                       tuple(r[f"seed{s}"] for s in seeds), float(r["tonic_hz"]),
                       FULL_AUDIO_DIR / r["audio"], float(r["seconds"])) for r in rows)


def videos(role, seed=0):
    return [v for v in load_videos() if v.role(seed) == role]


def by_id():
    return {v.video: v for v in load_videos()}


# ------------------------------------------------------------------ the whole-recording CQT


def cqt_path(video):
    return CQT_DIR / f"{video.video}.npy"


def build_cqt(video, force=False, chunk_seconds=600.0, pad_seconds=5.0):
    """Cache the recording's Sa-anchored CQT as uint8 decibels below its loudest bin.

    Computed in 10-minute chunks with 5 s of overlap: one call over an 84-minute recording
    materialises a multi-gigabyte STFT inside librosa, and chunking bounds that without
    changing a frame -- every chunk starts on a hop boundary, and the overlap is several
    times the longest CQT filter (~0.9 s at the lowest bin).
    """
    import librosa
    import soundfile as sf

    out_path = cqt_path(video)
    if out_path.exists() and not force:
        return out_path

    y, sr = sf.read(str(video.audio), dtype="float32", always_2d=True)   # read-only
    y = librosa.resample(y.mean(axis=1), orig_sr=sr, target_sr=audio.SR_CQT,
                         res_type="soxr_hq")
    hop = audio.CQT_HOP
    n_frames = 1 + len(y) // hop
    step = int(chunk_seconds * audio.SR_CQT) // hop * hop
    pad = int(pad_seconds * audio.SR_CQT) // hop * hop
    fmin = tonic_mod.anchor_fmin(video.tonic_hz)

    mag = np.zeros((N_BINS, n_frames), dtype=np.float32)
    for s0 in range(0, len(y), step):
        a, b = max(0, s0 - pad), min(len(y), s0 + step + pad)
        C = np.abs(librosa.cqt(y[a:b], sr=audio.SR_CQT, fmin=fmin, n_bins=N_BINS,
                               bins_per_octave=audio.CQT_BINS_PER_OCTAVE, hop_length=hop))
        f0, f1 = s0 // hop, min(n_frames, (s0 + step) // hop)
        j0 = (s0 - a) // hop
        mag[:, f0:f1] = C[:, j0:j0 + (f1 - f0)]

    db = 20.0 * np.log10(np.maximum(mag, 1e-10) / max(float(mag.max()), 1e-10))
    q = np.round((np.clip(db, -DB_RANGE, 0.0) + DB_RANGE) * (255.0 / DB_RANGE))
    audio._atomic_save(out_path, q.astype(np.uint8))
    return out_path


def window_features(q):
    """(144, 431) uint8 crop -> (144, 431) float32, normalised as `audio.cqt` does a clip."""
    db = q.astype(np.float32) * (DB_RANGE / 255.0) - DB_RANGE
    return (np.maximum(db - db.max(), -80.0) + 80.0) / 80.0


def usable_starts(q, trim_seconds=30.0, loud_fraction=0.8, loudness_db=20.0):
    """Frame positions a training window may start at. See the module docstring."""
    n = q.shape[1] - WINDOW_FRAMES + 1
    if n <= 0:
        return np.array([0])
    peak = q.max(axis=0).astype(np.float32) * (DB_RANGE / 255.0)
    loud = (peak >= np.median(peak) - loudness_db).astype(np.int64)
    cs = np.concatenate([[0], np.cumsum(loud)])
    frac = (cs[WINDOW_FRAMES:WINDOW_FRAMES + n] - cs[:n]) / WINDOW_FRAMES
    trim = int(trim_seconds * FRAMES_PER_SECOND)
    inside = np.zeros(n, dtype=bool)
    inside[trim:max(trim, n - trim)] = True
    for ok in (inside & (frac >= loud_fraction), inside, np.ones(n, dtype=bool)):
        if ok.any():                    # a short or quiet recording degrades, never empties
            return np.flatnonzero(ok)


class FullAudioCQTDataset(_Base):
    """20 s windows of the full recordings, as items shaped exactly like `CQTDataset`'s.

    train=True    each read draws a fresh random usable start, so `windows_per_video` is
                  a per-epoch count, not a fixed set
    fixed_pool    with train=True: train on the fixed set below instead, reused every epoch
                  (pitch jitter still random) -- what a dataset with `windows_per_video`
                  clips per performance would give. A recording too short for that many
                  distinct windows gets overlapping ones, as a dataset cut from it would.
    train=False   `windows_per_video` fixed starts per video, evenly spaced over the usable
                  positions -- the same every time, for evaluation

    `clips` holds one `data.Clip` per item (clip_id `<raag>/<video>@<k>`), so the metrics,
    the video vote and the confusion plots work on this dataset unchanged.
    """

    def __init__(self, videos, windows_per_video, train, freq_shift_bins=0,
                 trim_seconds=30.0, loud_fraction=0.8, loudness_db=20.0, fixed_pool=False):
        self.videos = list(videos)
        self.k = int(windows_per_video)
        self.fixed_pool = fixed_pool
        self.freq_shift_bins = freq_shift_bins
        missing = [v.video for v in self.videos if not cqt_path(v).exists()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} recordings have no cached CQT "
                                    f"(e.g. {missing[0]}); run scripts/04_build_fullaudio_cache.py")
        self.starts = [usable_starts(np.load(cqt_path(v), mmap_mode="r"), trim_seconds,
                                     loud_fraction, loudness_db) for v in self.videos]
        self.fixed = [s[np.linspace(0, len(s) - 1, self.k).round().astype(int)]
                      for s in self.starts]
        clips = [Clip(f"{v.raag}/{v.video}@{k}", v.raag, v.label, v.split, v.video, k,
                      v.tonic_hz) for v in self.videos for k in range(self.k)]
        super().__init__(clips, tonic="normalise", train=train)

    def __getitem__(self, i):
        vi, k = divmod(i, self.k)
        starts = self.starts[vi]
        fresh = self.train and not self.fixed_pool
        s = int(starts[np.random.randint(len(starts))] if fresh else self.fixed[vi][k])
        q = np.asarray(np.load(cqt_path(self.videos[vi]), mmap_mode="r")[:, s:s + WINDOW_FRAMES])
        if q.shape[1] < WINDOW_FRAMES:                 # a recording shorter than a window
            q = np.pad(q, ((0, 0), (0, WINDOW_FRAMES - q.shape[1])), constant_values=0)
        x = window_features(q)
        if self.train and self.freq_shift_bins:        # same jitter as CQTDataset
            x = np.roll(x, int(np.random.randint(-self.freq_shift_bins,
                                                 self.freq_shift_bins + 1)), axis=0)
        return self._wrap(torch.from_numpy(np.ascontiguousarray(x))[None].float(), i)
