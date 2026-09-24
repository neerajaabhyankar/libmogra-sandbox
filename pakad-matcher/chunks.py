"""Pick stretches to notate: some slow, some dense, spread over raags and recordings.

    poetry run python chunks.py

Density is held notes per second (the same `matcher._held` the scorer uses), so "alap" and
"taan" here mean what the model sees, not what a tempo tracker would say.
"""

import json

import numpy as np

import _bootstrap  # noqa: F401
import config as C
import fullaudio
import matcher


def density(ctr):
    """Held notes per second, per frame, over a rolling window."""
    held = matcher._held(ctr.cents, ctr.hop, C.MATCH)
    onsets = np.r_[False, ~held[:-1] & held[1:]].astype(float)
    w = int(round(C.CHUNK_ALAP_S / ctr.hop))
    kernel = np.ones(w) / (w * ctr.hop)
    return np.convolve(onsets, kernel, mode="same"), held


def pick(video, rng):
    ctr = fullaudio.contour(video)
    dens, _ = density(ctr)
    voiced = (~np.isnan(ctr.cents)).astype(float)
    out = []
    for tag, secs in (("alap", C.CHUNK_ALAP_S), ("taan", C.CHUNK_TAAN_S)):
        w = int(round(secs / ctr.hop))
        if len(ctr.cents) < 2 * w:
            continue
        frac = np.convolve(voiced, np.ones(w) / w, mode="same")
        ok = frac >= C.CHUNK_MIN_VOICED
        if not ok.any():
            continue
        d = np.where(ok, dens, np.nan)
        # the slowest / densest stretch, away from the edges
        edge = slice(w, len(d) - w)
        idx = np.arange(len(d))[edge]
        vals = d[edge]
        if np.isnan(vals).all():
            continue
        centre = int(idx[np.nanargmin(vals) if tag == "alap" else np.nanargmax(vals)])
        t0 = max(0.0, (centre - w // 2) * ctr.hop)
        out.append(dict(video=video, raag=fullaudio.index()[video].raag, kind=tag,
                        t0=round(t0, 2), t1=round(t0 + secs, 2),
                        notes_per_s=round(float(dens[centre]), 2)))
    return out


def build():
    C.CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(C.S3_SEED)
    chunks = []
    for raag in C.ANNOTATION_RAAGS:
        vids = fullaudio.cached_videos(tuple([raag]))
        for v in list(rng.permutation(vids))[:C.CHUNK_RECORDINGS_PER_RAAG]:
            chunks += pick(str(v), rng)
    for i, ch in enumerate(chunks):
        ch["id"] = f"{ch['raag']}_{ch['kind']}_{i:02d}"
        ch["tonic_hz"] = fullaudio.index()[ch["video"]].tonic_hz
        _snippet(ch)
    (C.S3_DIR / "chunks.json").write_text(json.dumps(chunks, indent=1))
    print(f"{len(chunks)} chunks -> {C.S3_DIR / 'chunks.json'}")
    for ch in chunks:
        print(f"  {ch['id']:26s} {ch['video']}  {ch['t0']:8.1f}-{ch['t1']:.1f}s  "
              f"{ch['notes_per_s']:.2f} notes/s")


def _snippet(ch):
    import librosa
    import soundfile as sf
    fa = fullaudio.index()[ch["video"]]
    y, sr = librosa.load(fa.path, sr=None, mono=True, offset=ch["t0"], duration=ch["t1"] - ch["t0"])
    sf.write(C.CHUNK_DIR / f"{ch['id']}.wav", y, sr)


if __name__ == "__main__":
    build()
