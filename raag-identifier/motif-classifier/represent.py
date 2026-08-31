"""Cached note events (+ a tonic choice) -> swar sequences.

The tonic is the whole ballgame (see plan.md), so it is a parameter here rather than a
fact: `tonic_mode` picks between the clip's own estimate, an estimate pooled over every
chunk of the same video, and leaving it to the matcher via `shift`.

Two corrections on top of melody-extraction's heuristic, both found necessary by looking at
the transcriptions (see plan.md "tuning offset"):

* the heuristic snaps the tonic to the nearest semitone **of the A440 grid**, so it can sit
  up to 50 cents off the actual Sa — which then biases *every* note toward the wrong swar.
  `tonic_refine` recovers that offset from the pitch histogram.
* the ±1 octave clamp threw away most of a clip whenever the tonic estimate landed an
  octave low. The default is now ±2 octaves.
"""

import sys
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MELODY_DIR = HERE.parent / "melody-extraction"
if str(MELODY_DIR) not in sys.path:
    sys.path.insert(0, str(MELODY_DIR))

from pipeline import estimate_tonic_hz  # noqa: E402  (melody-extraction's shared heuristic)
from note_segmentation import segment_notes  # noqa: E402

import _bootstrap  # noqa: F401  (puts raag-identifier/ on sys.path)
from utils.extract import load_cache, list_clips  # noqa: E402
from utils.raagdb import SWAR_NAMES  # noqa: E402


@dataclass(frozen=True)
class Params:
    """Everything about turning audio-derived notes into a swar string."""

    tracker: str = "tony"
    separate: str = None  # ../source-separation backend applied before tracking, or None
    note_source: str = "hmm"  # "hmm" = the tracker's own notes | "segment" = melody-extraction's segment_notes on the frame track
    tonic_mode: str = "chroma_video"  # {clip,video} = melody-extraction's heuristic; {chroma_clip,chroma_video} adds the Sa-vs-Pa correction; "true" = the dataset's hand annotation (v1 only)
    tonic_refine: bool = True  # recover the sub-semitone tuning offset
    min_dur: float = 0.0  # drop notes shorter than this (seconds)
    collapse_repeats: bool = True
    max_cents_dev: float = 2400.0  # drop notes further than this from the tonic
    quantize: str = "semitone"  # "semitone" = 12-TET rounding | "shruti" = libmogra's uneven SWAR_BOUNDARIES
    shift: int = 0  # semitones added to every note (only used for one-off inspection)
    # chroma_* tonic modes only: how Sa is told apart from Pa
    tonic_alpha: float = 0.6  # reward for a strong Pa a fifth above the candidate
    tonic_beta: float = 0.9  # penalty for the candidate itself being a fifth above a strong class
    tonic_gamma: float = 0.05  # penalty for the sung range sitting wrong relative to the candidate
    tonic_median_target: float = 6.0  # expected median sung pitch, semitones above Sa
    # only for note_source="segment"
    seg_tol_cents: float = 50.0
    seg_min_note_dur: float = 0.2


@dataclass
class Clip:
    clip_id: str
    raag: str
    split: str
    video: str
    swars: list  # list[int], 0..11 (hard, nearest-bin)
    durs: list  # list[float], seconds, parallel to swars
    octaves: list  # list[int], parallel to swars (0 = tonic's octave)
    tonic_hz: float
    cents: list = None  # continuous cents vs tonic, octave-folded to [0,1200) — the
    #                     un-quantized truth the swar index is a lossy read of


# ---------------------------------------------------------------- tonic


def refine_tonic(f0_hz, voiced, tonic_hz, max_cents_dev=2400.0):
    """Recover the sub-semitone offset the coarse heuristic threw away.

    Voiced frames within `max_cents_dev` of the tonic are folded onto one semitone
    (cents mod 100) and averaged *circularly* — the resulting offset is how far the whole
    performance sits from the A440 grid the heuristic snapped to. Shifting the tonic by it
    re-centres every note on its swar instead of leaving them all biased one way.
    """
    f0 = f0_hz[voiced & (f0_hz > 0)]
    if len(f0) < 50:
        return tonic_hz
    cents = 1200.0 * np.log2(f0 / tonic_hz)
    cents = cents[np.abs(cents) <= max_cents_dev]
    if len(cents) < 50:
        return tonic_hz
    theta = 2 * np.pi * (cents % 100.0) / 100.0
    offset = np.angle(np.exp(1j * theta).mean()) / (2 * np.pi) * 100.0  # in (-50, 50]
    return tonic_hz * 2.0 ** (offset / 1200.0)


def chroma_tonic(f0_hz, voiced, tonic_hz, alpha=0.6, beta=0.9, gamma=0.05, median_target=6.0,
                 max_cents_dev=2400.0):
    """Pick which of the 12 semitones around the coarse estimate is actually Sa.

    `estimate_tonic_hz` takes the mode of the voiced-pitch distribution, and in this corpus
    that mode is very often the **tanpura's Pa**, not Sa: the diagnostics show the correct
    rotation peaking at +7 semitones far more than chance. Two cues separate Sa from Pa,
    both read off the same frame track melody-extraction already produced:

      * a Sa has a strong Pa a fifth above it, and is *not* itself a fifth above another
        strong pitch class  ->  h[c] + alpha*h[c+7] - beta*h[c-7]
      * Sa sits near the bottom of the sung range — the median sung pitch is roughly
        mid-madhya-saptak, a few semitones above Sa, so a candidate the performance mostly
        sits *below* is the wrong one  ->  -gamma * |median - c - median_target|

    Returns the corrected tonic in Hz.
    """
    f0 = f0_hz[voiced & (f0_hz > 0)]
    if len(f0) < 50:
        return tonic_hz
    cents = 1200.0 * np.log2(f0 / tonic_hz)
    cents = cents[np.abs(cents) <= max_cents_dev]
    if len(cents) < 50:
        return tonic_hz

    semis = np.round(cents / 100.0).astype(int)
    h = np.bincount(semis % 12, minlength=12).astype(float)
    h /= h.sum()

    median = float(np.median(semis))
    best_k, best_s = 0, -np.inf
    for k in range(-6, 6):  # candidate Sa = k semitones from the coarse estimate
        s = (
            h[k % 12]
            + alpha * h[(k + 7) % 12]
            - beta * h[(k - 7) % 12]
            - gamma * abs(median - k - median_target)
        )
        if s > best_s:
            best_k, best_s = k, s
    return tonic_hz * 2.0 ** (best_k / 12.0)


def _apply_tonic_policy(f0, voiced, mode, refine, max_cents_dev, chroma_kw):
    t = estimate_tonic_hz(f0, voiced)
    if mode.startswith("chroma"):
        t = chroma_tonic(f0, voiced, t, max_cents_dev=max_cents_dev, **chroma_kw)
    if refine:
        t = refine_tonic(f0, voiced, t, max_cents_dev)
    return t


def _video_tonics(cache, clips, mode, refine, max_cents_dev, chroma_kw):
    """One tonic per video, from the frame tracks of all its chunks concatenated.

    Chunks of a video are the same performer in the same recording, and they sit side by
    side in the dataset, so pooling them is available at test time too — it just gives the
    mode-of-pitch heuristic 3x the audio to work with.
    """
    by_video = {}
    for c in clips:
        if c["clip_id"] in cache:
            by_video.setdefault(c["video"], []).append(c["clip_id"])
    tonics = {}
    for video, ids in by_video.items():
        f0 = np.concatenate([cache[i]["f0"] for i in ids])
        voiced = np.concatenate([cache[i]["voiced"] for i in ids])
        tonics[video] = _apply_tonic_policy(f0, voiced, mode, refine, max_cents_dev, chroma_kw)
    return tonics


def _clip_tonics(cache, mode, refine, max_cents_dev, chroma_kw):
    return {
        k: _apply_tonic_policy(v["f0"], v["voiced"], mode, refine, max_cents_dev, chroma_kw)
        for k, v in cache.items()
    }


def _annotated_tonics(clips):
    """The dataset's hand-annotated Sa, used verbatim.

    No refinement: the annotation was already snapped to a peak of the recording's own
    pitch histogram, so `refine_tonic` would only let a bad tracker pull it off truth.
    """
    missing = [c["clip_id"] for c in clips if c.get("true_tonic_hz") is None]
    if missing:
        raise ValueError(
            f"tonic_mode='true' needs the v1 annotation; {len(missing)} clips lack it "
            f"(first: {missing[0]}). Is RAAG_DATA_VERSION set to v0?"
        )
    per_clip = {c["clip_id"]: c["true_tonic_hz"] for c in clips}
    per_video = {c["video"]: c["true_tonic_hz"] for c in clips}
    return per_clip, per_video


@lru_cache(maxsize=16)
def _load(tracker, mode, refine, max_cents_dev, chroma_items, separate=None):
    """Cache + both tonic estimates, computed once per (tracker, tonic setting)."""
    cache = load_cache(tracker, separate)
    chroma_kw = dict(chroma_items)
    clips = [c for c in list_clips() if c["clip_id"] in cache]
    if mode == "true":
        clip_t, video_t = _annotated_tonics(clips)
    else:
        clip_t = _clip_tonics(cache, mode, refine, max_cents_dev, chroma_kw)
        video_t = _video_tonics(cache, clips, mode, refine, max_cents_dev, chroma_kw)
    return cache, clips, clip_t, video_t


# ---------------------------------------------------------------- notes -> swars


def _notes_from_frames(entry, tonic_hz, p: Params):
    """melody-extraction's segment_notes on the cached frame track -> (N,3) [t0,t1,hz]."""
    f0, voiced, hop = entry["f0"], entry["voiced"], entry["hop"]
    with np.errstate(divide="ignore", invalid="ignore"):
        cents = 1200.0 * np.log2(np.clip(f0, 1e-6, None) / tonic_hz)
    segs = segment_notes(cents, voiced, hop, tol_cents=p.seg_tol_cents, min_note_dur=p.seg_min_note_dur)
    return np.array(
        [[s.t_start, s.t_end, tonic_hz * 2.0 ** (s.cents_relative / 1200.0)] for s in segs],
        dtype=np.float32,
    ).reshape(-1, 3)


def notes_to_swars(notes, tonic_hz, p: Params):
    """notes: (N,3) [t_start, t_end, f0_hz]. Returns (swars, durs, octaves)."""
    if len(notes) == 0:
        return [], [], [], []
    dur = notes[:, 1] - notes[:, 0]
    hz = notes[:, 2]
    keep = (hz > 0) & (dur >= p.min_dur)
    if not keep.any():
        return [], [], [], []
    cents = 1200.0 * np.log2(hz[keep] / tonic_hz) + 100.0 * p.shift
    dur = dur[keep]

    keep2 = np.abs(cents) <= p.max_cents_dev
    cents, dur = cents[keep2], dur[keep2]
    if len(cents) == 0:
        return [], [], [], []

    if p.quantize == "shruti":
        # libmogra's SWAR_BOUNDARIES are deliberately *not* at the 12-TET midpoints — komal
        # swars in the Bhairav/Todi families sit sharp of their equal-tempered position, and
        # rounding at 50 cents sends them to the wrong swar. Use the DB's own boundaries.
        semis = _shruti_bin(cents)
    else:
        semis = np.round(cents / 100.0).astype(int)
    swars = (semis % 12).tolist()
    octaves = np.floor_divide(semis, 12).tolist()
    durs = dur.tolist()
    folded = (cents % 1200.0).tolist()  # continuous chroma, kept alongside the hard bin

    if p.collapse_repeats:
        s2, d2, o2, c2 = [], [], [], []
        for s, d, o, c in zip(swars, durs, octaves, folded):
            if s2 and s2[-1] == s:
                # duration-weighted circular mean, so a merged run keeps its true centre
                # rather than jumping to whichever note came last
                w = d2[-1] + d
                prev = np.deg2rad(c2[-1] * 360.0 / 1200.0)
                cur = np.deg2rad(c * 360.0 / 1200.0)
                ang = np.angle(d2[-1] * np.exp(1j * prev) + d * np.exp(1j * cur))
                c2[-1] = float((np.rad2deg(ang) % 360.0) * 1200.0 / 360.0)
                d2[-1] = w
            else:
                s2.append(s)
                d2.append(d)
                o2.append(o)
                c2.append(c)
        swars, durs, octaves, folded = s2, d2, o2, c2
    return swars, durs, octaves, folded


_SHRUTI_EDGES = None


def _shruti_bin(cents):
    """Octave-relative cents -> swar index using libmogra's uneven SWAR_BOUNDARIES."""
    global _SHRUTI_EDGES
    if _SHRUTI_EDGES is None:
        from libmogra.datatypes import SWAR_BOUNDARIES

        _SHRUTI_EDGES = 1200.0 * np.log2(np.asarray(SWAR_BOUNDARIES, dtype=float))
    octave = np.floor(cents / 1200.0).astype(int)
    within = cents - 1200.0 * octave
    return octave * 12 + np.searchsorted(_SHRUTI_EDGES, within) % 12


def build_clips(p: Params):
    """All clips as Clip objects under representation params `p`."""
    cache, clip_meta, clip_tonics, video_tonics = _load(
        p.tracker,
        p.tonic_mode,
        p.tonic_refine,
        p.max_cents_dev,
        (
            ("alpha", p.tonic_alpha),
            ("beta", p.tonic_beta),
            ("gamma", p.tonic_gamma),
            ("median_target", p.tonic_median_target),
        ),
        p.separate,
    )
    out = []
    for c in clip_meta:
        cid = c["clip_id"]
        per_clip = p.tonic_mode in ("clip", "chroma_clip")  # "true" is per-video by construction
        tonic = clip_tonics[cid] if per_clip else video_tonics[c["video"]]
        notes = cache[cid]["notes"] if p.note_source == "hmm" else _notes_from_frames(cache[cid], tonic, p)
        swars, durs, octaves, folded = notes_to_swars(notes, tonic, p)
        out.append(
            Clip(
                clip_id=cid,
                raag=c["raag"],
                split=c["split"],
                video=c["video"],
                swars=swars,
                durs=durs,
                octaves=octaves,
                tonic_hz=tonic,
                cents=folded,
            )
        )
    return out


def shifted(clip: Clip, k: int) -> Clip:
    """The same clip with every swar rotated by k semitones (tonic search)."""
    if k == 0:
        return clip
    return replace(
        clip,
        swars=[(s + k) % 12 for s in clip.swars],
        octaves=[o + (s + k) // 12 for s, o in zip(clip.swars, clip.octaves)],
        tonic_hz=clip.tonic_hz * 2.0 ** (-k / 12.0),
    )


def to_string(swars):
    return "".join(SWAR_NAMES[s] for s in swars)


if __name__ == "__main__":
    for src in ("hmm", "segment"):
        p = Params(note_source=src)
        clips = build_clips(p)
        lens = [len(c.swars) for c in clips]
        print(f"--- note_source={src}: {len(clips)} clips, swars/clip "
              f"min {min(lens)} med {int(np.median(lens))} max {max(lens)}, "
              f"{sum(l < 2 for l in lens)} degenerate")
        for c in clips[:5]:
            print(f"  {c.clip_id[:48]:48s} {c.tonic_hz:7.1f}Hz  {to_string(c.swars)[:64]}")
