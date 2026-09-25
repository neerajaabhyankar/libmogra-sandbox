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


def replace(ids):
    """Swap out chunks that did not work out, for the same raag and kind from another recording."""
    chunks = json.loads((C.S3_DIR / "chunks.json").read_text())
    by_id = {c["id"]: c for c in chunks}
    used = {c["video"] for c in chunks}
    rng = np.random.default_rng(C.S3_SEED + 1)
    for cid in ids:
        old = by_id[cid]
        others = [v for v in fullaudio.cached_videos(tuple([old["raag"]]))
                  if v not in used and v not in reserved_videos()]
        if not others:
            print(f"{cid}: no unused recording of {old['raag']} left"); continue
        v = str(rng.permutation(others)[0])
        fresh = [c for c in pick(v, rng) if c["kind"] == old["kind"]]
        if not fresh:
            print(f"{cid}: nothing suitable in {v}"); continue
        ch = fresh[0]
        ch["id"] = f"{ch['raag']}_{ch['kind']}_{len(chunks):02d}"
        ch["tonic_hz"] = fullaudio.index()[v].tonic_hz
        ch["replaces"] = cid
        _snippet(ch)
        chunks = [c for c in chunks if c["id"] != cid] + [ch]
        used.add(v)
        (C.CHUNK_DIR / f"{cid}.wav").unlink(missing_ok=True)
        print(f"{cid} -> {ch['id']}  ({v}, {ch['t0']:.0f}-{ch['t1']:.0f}s, {ch['notes_per_s']} notes/s)")
    chunks.sort(key=lambda c: c["id"])
    (C.S3_DIR / "chunks.json").write_text(json.dumps(chunks, indent=1))


def reserved_videos():
    """Recordings the test set sits on. Training notation must not touch them."""
    import config as C
    out = set()
    labels = C.S3_DIR / "labels.jsonl"
    if labels.exists():
        for line in open(labels):
            r = json.loads(line)
            if r.get("video"):
                out.add(r["video"])
    return out


def build(raags=None):
    """Add chunks for `raags`, keeping every chunk that already exists.

    Existing chunks are never rewritten: `annotations/notations.jsonl` refers to them by id, and
    their `video`/`t0` is what turns a stretch's times into recording times. Recordings already
    used -- by another chunk, or by any judgment -- are skipped, so training and test stay apart.
    """
    C.CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(C.S3_SEED)
    path = C.S3_DIR / "chunks.json"
    chunks = json.loads(path.read_text()) if path.exists() else []
    taken = {c["video"] for c in chunks} | reserved_videos()
    next_i = max([int(c["id"].rsplit("_", 1)[1]) for c in chunks], default=-1) + 1

    added = []
    for raag in (raags or C.ANNOTATION_RAAGS):
        if raag in getattr(C, "TEST_ONLY_RAAGS", []):
            print(f"{raag}: test-only, not notated"); continue
        vids = [v for v in fullaudio.cached_videos(tuple([raag])) if v not in taken]
        if not vids:
            print(f"{raag}: no free recordings (all judged or already chunked)"); continue
        for v in list(rng.permutation(vids))[:C.CHUNK_RECORDINGS_PER_RAAG]:
            taken.add(str(v))
            for ch in pick(str(v), rng):
                ch["id"] = f"{ch['raag']}_{ch['kind']}_{next_i:02d}"
                ch["tonic_hz"] = fullaudio.index()[str(v)].tonic_hz
                next_i += 1
                _snippet(ch)
                added.append(ch)
    path.write_text(json.dumps(chunks + added, indent=1))
    print(f"\n{len(added)} chunks added, {len(chunks) + len(added)} in total -> {path}")
    for ch in added:
        print(f"  {ch['id']:26s} {ch['video']}  {ch['t0']:8.1f}-{ch['t1']:.1f}s  "
              f"{ch['notes_per_s']:.2f} notes/s")


def _snippet(ch):
    import librosa
    import soundfile as sf
    fa = fullaudio.index()[ch["video"]]
    y, sr = librosa.load(fa.path, sr=None, mono=True, offset=ch["t0"], duration=ch["t1"] - ch["t0"])
    sf.write(C.CHUNK_DIR / f"{ch['id']}.wav", y, sr)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--replace", nargs="+", default=None, help="chunk ids to swap out")
    ap.add_argument("--raags", nargs="+", default=None, help="raags to add chunks for")
    a = ap.parse_args()
    replace(a.replace) if a.replace else build(a.raags)
