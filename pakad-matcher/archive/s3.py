"""S3: build a blind annotation pool per phrase, play it, record y/n answers.

A positive is an ornamented path that still traces the phrase -- something Neeraja would
notate as that phrase. A negative is an approximate presence that cannot be identified as
it (too much ornamentation, or a different phrase). Own raag only: we use the raag as a
searching ground, not as evidence.

    poetry run python s3.py build                       # pool + wavs for every phrase
    poetry run python s3.py annotate --phrase Bageshree#0     # the interactive loop
    poetry run python s3.py play   --phrase Bageshree#0 --index 1 [--repeat 2]
    poetry run python s3.py record --phrase Bageshree#0 --index 1 --answers y
    poetry run python s3.py report
"""

import argparse
import json
import subprocess
from datetime import datetime, timezone

import numpy as np

import _bootstrap  # noqa: F401
import config as C
import contour
import matcher
import mukhyangas

BATCH = 6
LABELS = C.S3_DIR / "labels.jsonl"


def pool_path(p):
    return C.S3_DIR / "pool" / f"{p.slug}.json"


def audio_path(p, i):
    return C.S3_DIR / "audio" / p.slug / f"{i:02d}.wav"


def build_pool(p, rng):
    """Candidates from the phrase's own raag, sampled across cost bands, ≤1 per clip."""
    cands = []
    max_dur = C.S3_MAX_S_PER_NOTE * len(p.swars)
    for cid in sorted(c for c in contour.cached_ids() if contour.clips()[c].raag == p.raag):
        ctr = contour.contour(cid)
        for c in matcher.match(ctr, p.swars, top_k=C.S3_TOP_PER_CLIP):
            if c.t1 - c.t0 > max_dur:                      # a stretched-out match
                continue
            n_held = matcher.held_notes(ctr.cents[c.f0:c.f1 + 1], ctr.hop)
            if n_held > len(p.swars) + C.S3_EXTRA_NOTES:   # a taan, not a phrase
                continue
            c.n_held = n_held
            cands.append((c.cost, cid, c))
    cands.sort(key=lambda t: t[0])
    costs = np.array([c for c, _, _ in cands])

    chosen, used_clip, per_video = [], set(), {}

    def take(idx, n, band):
        got = 0
        for i in idx:
            cost, cid, c = cands[i]
            v = contour.clips()[cid].video
            if cid in used_clip or per_video.get(v, 0) >= C.S3_MAX_PER_VIDEO:
                continue
            used_clip.add(cid); per_video[v] = per_video.get(v, 0) + 1
            chosen.append(dict(clip_id=cid, video=v, t0=round(c.t0, 2), t1=round(c.t1, 2),
                               cost=round(c.cost, 3), pitch_cost=round(c.pitch_cost, 3),
                               orn_frac=round(c.orn_frac, 3), leaps=c.leaps,
                               n_held=c.n_held, dur=round(c.t1 - c.t0, 2), band=band))
            got += 1
            if got == n:
                break
        return got

    for name, lo, hi, n in C.S3_BANDS:
        idx = list(np.flatnonzero((costs >= lo) & (costs < hi)))
        rng.shuffle(idx)
        take(idx, n, name)
    # short of a full pool (a phrase the matcher rarely fits): fall back to the next best
    if len(chosen) < C.S3_POOL_PER_PHRASE:
        cap = C.S3_BANDS[-1][2]
        take(list(np.flatnonzero(costs < cap)), C.S3_POOL_PER_PHRASE - len(chosen), "fallback")
    order = rng.permutation(len(chosen))           # presented blind: no cost ordering
    return [chosen[i] for i in order]


def build(raags=None):
    rng = np.random.default_rng(C.S3_SEED)
    (C.S3_DIR / "pool").mkdir(parents=True, exist_ok=True)
    for p in mukhyangas.load(raags=raags):
        items = build_pool(p, rng)
        pool_path(p).write_text(json.dumps(
            dict(phrase_id=p.id, raag=p.raag, phrase=p.text, source=p.source,
                 matcher=C.MATCHER_VERSION, context_s=C.S3_CONTEXT_S, items=items), indent=1))
        (C.S3_DIR / "audio" / p.slug).mkdir(parents=True, exist_ok=True)
        for i, it in enumerate(items):
            _write_snippet(contour.clips()[it["clip_id"]], it["t0"], it["t1"], audio_path(p, i))
        print(f"{p.id:22s} {p.text:16s} {len(items)} candidates  "
              f"cost {min(i['cost'] for i in items):.2f}–{max(i['cost'] for i in items):.2f}")


def _write_snippet(clip, t0, t1, path):
    import librosa
    import soundfile as sf
    pad = C.S3_CONTEXT_S
    y, sr = librosa.load(clip.path, sr=None, mono=True,
                         offset=max(0.0, t0 - pad), duration=(t1 - t0) + 2 * pad)
    sf.write(path, np.concatenate([y, np.zeros(int(C.S3_TAIL_S * sr), y.dtype)]), sr)


def _phrase(pid):
    for p in mukhyangas.load(only_annotate=False):
        if p.id == pid:
            return p
    raise SystemExit(f"unknown phrase {pid}")


def play(pid, index=None, batch=None, repeat=1):
    """One candidate (--index, 1-based as presented) or a whole batch."""
    p = _phrase(pid)
    items = json.loads(pool_path(p).read_text())["items"]
    idx = [index - 1] if index else range(batch * BATCH, min((batch + 1) * BATCH, len(items)))
    for i in idx:
        for _ in range(repeat):
            subprocess.run(["afplay", str(audio_path(p, i))], check=True)
    print(f"{p.id} {p.text}: played {[i + 1 for i in idx]} of {len(items)}")


def record(pid, answers, index=None, batch=None, annotator="neeraja"):
    p = _phrase(pid)
    items = json.loads(pool_path(p).read_text())["items"]
    ans = answers.replace(",", " ").split()
    lo = (index - 1) if index else batch * BATCH
    n = 1 if index else min(BATCH, len(items) - lo)
    if len(ans) != n:
        raise SystemExit(f"expected {n} answers, got {len(ans)}")
    _write([(lo + j, {"y": "yes", "n": "no", "u": "unsure"}[a.lower()[0]])
            for j, a in enumerate(ans)], p, items, annotator)
    left = [i for i in range(len(items)) if i > lo + n - 1]
    print(f"{p.id}: recorded {' '.join(ans)} for #{lo + 1}"
          f"{'-' + str(lo + n) if n > 1 else ''}; {len(left)} left")


def annotate(pid, redo=False, annotator="neeraja"):
    """Interactive loop: play a candidate, take a verdict, move on. Resumes where you left off."""
    p = _phrase(pid)
    items = json.loads(pool_path(p).read_text())["items"]
    done = {r["index"] for r in _labels() if r["phrase_id"] == p.id} if not redo else set()
    todo = [i for i in range(len(items)) if i not in done]
    print(f"\n{p.id}   {p.text}   ({p.source})")
    print(f"{len(todo)} of {len(items)} left"
          f"{'' if not done else f' ({len(done)} already judged)'}")
    print("  y = yes, this is the phrase   n = no   u = unsure")
    print("  r = replay   rr = replay twice   s = skip   q = quit (progress is saved)\n")
    for k, i in enumerate(todo, 1):
        print(f"[{k}/{len(todo)}]  candidate {i + 1}", flush=True)
        subprocess.run(["afplay", str(audio_path(p, i))], check=True)
        while True:
            a = input("   y/n/u/r/s/q > ").strip().lower()
            if a in ("r", "rr"):
                for _ in range(1 if a == "r" else 2):
                    subprocess.run(["afplay", str(audio_path(p, i))], check=True)
                continue
            if a == "q":
                return print("\nstopped. rerun the same command to continue.")
            if a == "s":
                break
            if a and a[0] in "ynu":
                _write([(i, {"y": "yes", "n": "no", "u": "unsure"}[a[0]])], p, items, annotator)
                break
            print("   ? answer y, n, u, r, rr, s or q")
    print(f"\ndone with {p.id}.")
    report(p.id)


def _labels():
    return [json.loads(l) for l in open(LABELS)] if LABELS.exists() else []


def _write(pairs, p, items, annotator):
    C.S3_DIR.mkdir(parents=True, exist_ok=True)
    with open(LABELS, "a") as fh:
        for i, verdict in pairs:
            fh.write(json.dumps(dict(phrase_id=p.id, phrase=p.text, raag=p.raag, index=i,
                                     verdict=verdict, annotator=annotator,
                                     matcher=C.MATCHER_VERSION,
                                     ts=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                     **items[i])) + "\n")


def report(only=None):
    rows = [r for r in _labels() if only is None or r["phrase_id"] == only]
    if not rows:
        return print("no labels yet")
    seen = {}
    for r in rows:                                  # last label for a candidate wins
        seen[(r["phrase_id"], r["index"])] = r
    rows = list(seen.values())
    print(f"{len(rows)} labels over {len({r['phrase_id'] for r in rows})} phrases\n")
    print(f"{'phrase':24s} {'n':>3s} {'yes':>4s} {'no':>4s} {'?':>3s}   yes-rate by band")
    for pid in sorted({r["phrase_id"] for r in rows}):
        rs = [r for r in rows if r["phrase_id"] == pid]
        by = {b: [r for r in rs if r["band"] == b] for _, b, _, _ in
              [(0, b, 0, 0) for b, *_ in C.S3_BANDS]}
        bands = "  ".join(f"{b}: {np.mean([r['verdict'] == 'yes' for r in v]):.2f} ({len(v)})"
                          for b, v in by.items() if v)
        print(f"{pid:24s} {len(rs):3d} {sum(r['verdict']=='yes' for r in rs):4d} "
              f"{sum(r['verdict']=='no' for r in rs):4d} {sum(r['verdict']=='unsure' for r in rs):3d}   {bands}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--raags", nargs="+", default=None)
    for name in ("play", "record"):
        s = sub.add_parser(name)
        s.add_argument("--phrase", required=True)
        s.add_argument("--index", type=int, default=None, help="1-based, as presented")
        s.add_argument("--batch", type=int, default=None)
        s.add_argument("--answers" if name == "record" else "--repeat",
                       **({"required": True} if name == "record" else {"type": int, "default": 1}))
    an = sub.add_parser("annotate")
    an.add_argument("--phrase", required=True)
    an.add_argument("--redo", action="store_true")
    rp = sub.add_parser("report"); rp.add_argument("--phrase", default=None)
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.raags)
    elif a.cmd == "play":
        play(a.phrase, a.index, 0 if a.batch is None and not a.index else a.batch, a.repeat)
    elif a.cmd == "record":
        record(a.phrase, a.answers, a.index, a.batch)
    elif a.cmd == "annotate":
        annotate(a.phrase, a.redo)
    else:
        report(a.phrase)
