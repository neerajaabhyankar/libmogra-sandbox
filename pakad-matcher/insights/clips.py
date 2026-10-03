"""Insight clips: 30 s madhya-lay stretches, one per raag per split, for Neeraja to annotate.

    poetry run python -m insights.clips --build     # add missing clips -> annotations/insight_clips*
    poetry run python -m insights.clips --check     # the split rules below
    poetry run python -m insights.clips --report    # machine findings -> results/insights/clips.md

Annotate at http://localhost:8765/insights (annotate_app.py): per swar, aarohi / avarohi /
both *as it appears in this clip*; nyas = windows marked on the pitch track, each with its swar.

Split rules (checked by --check):
  I-R1  no recording appears in two splits
  I-R2  test and validation never use a notated recording (the functions are tuned on notation)
  I-R3  a train clip never overlaps a notated stretch
  I-R4  the registry is append-only: existing clips are never moved or re-cut; a bad one is
        marked `excluded` with the reason (e.g. Multani_test, wrong tonic) and left out

"Madhya" = the 30 s whose density (held notes per second) is nearest the recording's median.

Terms: [DATA.md § Glossary](../DATA.md#glossary).
"""

import argparse
import json

import numpy as np

import _bootstrap  # noqa: F401
import chunks
import config as C
import fullaudio
from insights import core
from utils import raagdb

SEED = C.S3_SEED + 3
REPORT = C.INSIGHTS_DIR / "clips.md"


def registry(with_excluded=False):
    """The clips; ones marked `excluded` (with the reason) are kept on file but left out."""
    clips = json.loads(C.INSIGHT_CLIPS.read_text()) if C.INSIGHT_CLIPS.exists() else []
    return clips if with_excluded else [c for c in clips if not c.get("excluded")]


def notated():
    """{video: [(t0, t1)]} of notation chunks."""
    out = {}
    for c in json.loads((C.S3_DIR / "chunks.json").read_text()):
        out.setdefault(c["video"], []).append((c["t0"], c["t1"]))
    return out


def build():
    C.INSIGHT_CLIP_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    clips, nota = registry(with_excluded=True), notated()
    split_of = {c["video"]: c["split"] for c in clips}
    have = {(c["split"], c["raag"]) for c in clips}          # an excluded clip is not replaced
    for split, raags in C.INSIGHT_CLIP_RAAGS.items():
        for raag in raags:
            if (split, raag) in have:
                continue
            vids = [v for v in fullaudio.cached_videos(tuple([raag]))
                    if v not in C.BAD_TONIC_VIDEOS and split_of.get(v, split) == split
                    and (split == "train" or v not in nota)]
            unused = [v for v in vids if v not in split_of]
            order = list(rng.permutation(unused)) or list(rng.permutation(vids))
            for v in map(str, order):
                got = chunks.pick(v, rng, kinds=("madhya",), avoid=nota.get(v, ()),
                                  secs=C.INSIGHT_CLIP_S)
                if got:
                    ch = dict(got[0], id=f"{raag}_{split}", split=split,
                              tonic_hz=fullaudio.index()[v].tonic_hz)
                    chunks._snippet(ch, C.INSIGHT_CLIP_DIR)
                    clips.append(ch); split_of[v] = split
                    print(f"  + {ch['id']:28s} {v}  {ch['t0']:7.1f}s  {ch['notes_per_s']:.2f} notes/s")
                    break
            else:
                print(f"  {raag} ({split}): no usable recording")
    C.INSIGHT_CLIPS.write_text(json.dumps(clips, indent=1))
    print(f"{len(clips)} clips -> {C.INSIGHT_CLIPS}")


def check():
    clips, nota, ok = registry(with_excluded=True), notated(), True
    by_video = {}
    for c in clips:
        by_video.setdefault(c["video"], set()).add(c["split"])
    for v, s in by_video.items():
        if len(s) > 1:
            print(f"I-R1 FAIL {v} in {sorted(s)}"); ok = False
    for c in clips:
        if c["split"] != "train" and c["video"] in nota:
            print(f"I-R2 FAIL {c['id']} is on a notated recording"); ok = False
        if any(a < c["t1"] and c["t0"] < b for a, b in nota.get(c["video"], ())):
            print(f"I-R3 FAIL {c['id']} overlaps a notated stretch"); ok = False
    counts = {s: sum(c["split"] == s and not c.get("excluded") for c in clips)
              for s in C.INSIGHT_CLIP_RAAGS}
    gone = [c["id"] for c in clips if c.get("excluded")]
    print(f"insight clips: {counts}  excluded {gone}  rules {'OK' if ok else 'FAILED'}")
    return ok


def labels():
    """Neeraja's answers, last per clip wins."""
    out = {}
    if C.INSIGHT_LABELS.exists():
        for line in open(C.INSIGHT_LABELS):
            r = json.loads(line)
            out[r["clip_id"]] = r
    return out


def db_row(raag):
    e = raagdb.RAAG_DB[raagdb.dataset_raags([raag])[raag].key]
    return dict(aaroha=" ".join(e.get("aaroha", [])), avaroha=" ".join(e.get("avaroha", [])),
                nyas=" ".join(sorted(set(e.get("aarohi_nyas", []) + e.get("avarohi_nyas", [])))))


def report(split="test"):
    """Machine findings per clip, for eyeballing. Not shown in the annotation app."""
    lines = [f"# Insight functions -- {split} clips", "",
             f"Settings: `config.INSIGHTS`. aarohi = after it, moves up >= "
             f"{C.INSIGHTS['dir_ratio']:g}x as often as down (>= {C.INSIGHTS['dir_min_count']} moves); "
             f"nyas = precedes >= {C.INSIGHTS['nyas_min_share']:.0%} of pauses "
             f"(>= {C.INSIGHTS['nyas_min_count']}). Times are seconds into the clip.", ""]
    for ch in [c for c in registry() if c["split"] == split]:
        ctr = fullaudio.contour(ch["video"])
        a, b = int(round(ch["t0"] / ctr.hop)), int(round(ch["t1"] / ctr.hop))
        ins = core.insights(ctr.cents[a:b], ctr.hop, wav=C.INSIGHT_CLIP_DIR / f"{ch['id']}.wav")
        db = db_row(ch["raag"])
        mv = "  ".join(f"{s} {m['up']}↑{m['down']}↓" for s, m in ins["moves"].items())
        pa = "  ".join(f"{s} {c}" for s, c in ins["pauses_after"].items())
        lines += [f"## {ch['raag']} -- `{ch['id']}.wav`",
                  f"{ch['video']} at {ch['t0']:.0f}–{ch['t1']:.0f} s, {ch['notes_per_s']} notes/s", "",
                  f"- **aarohi:** {' '.join(ins['aarohi']) or '—'}   **avarohi:** "
                  f"{' '.join(ins['avarohi']) or '—'}",
                  f"- **nyas:** {' '.join(ins['nyas']) or '—'}",
                  f"- moves after each swar: {mv or '—'}",
                  f"- swar before each of {ins['n_pauses']} pauses: {pa or '—'}",
                  "- pauses at: " + ", ".join(f"{t:.1f} {s}" for s, t in ins["pause_times"]),
                  f"- DB (hint only): aaroha `{db['aaroha']}`, avaroha `{db['avaroha']}`, "
                  f"nyas `{db['nyas'] or '—'}`", ""]
        print(f"{ch['raag']:15s} aarohi {ins['aarohi']} avarohi {ins['avarohi']} nyas {ins['nyas']}",
              flush=True)
    REPORT.write_text("\n".join(lines))
    print(f"-> {REPORT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.build:
        build()
    if a.build or a.check:
        check()
    if a.report:
        report()
