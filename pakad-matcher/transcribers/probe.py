"""Probes: ../transcriber's adaptation head on frozen feature blocks, trained on the notation,
scored held out by recording (../transcriber/plan.md § Adaptation).

    poetry run python -m transcribers.probe cqt bp melodia bp+melodia     # one config per argument

A config names feature blocks joined by "+": `cqt`, `bp` (../transcriber/features.py) and
`melodia` (Melodia's track from this repo's cache, as a block). Labels per 10 ms frame of every
notated chunk: the notated swar's pitch from Sa (equal temperament, with its octave) on the
note's frames -- what Neeraja heard, independent of any tracker, so Melodia is not favoured --
silent outside every notated stretch (with a margin), no label elsewhere. Within 50 cents of
that pitch = the right swar. Folds: corpus.folds() -- no recording on both sides. Baselines on the same frames:
Melodia itself and the basic_pitch source. Results: transcribers/probe/crossval.json.
"""

import json
import sys

import numpy as np

import config as C
import corpus
import fullaudio
from transcribers import source  # noqa: F401  (puts ../ on sys.path)
from transcriber import adapt, features  # noqa: E402
from utils import raagdb

HOP = features.HOP
SILENT_MARGIN_S = 0.1      # silent labels keep this far from any notated stretch
AGREE_CENTS = 50.0         # a frame counts as right within this of the label
OUT = C.TRANSCRIBERS_DIR / "probe" / "crossval.json"


def labels(rec, tonic, n):
    """(hz, mask) per frame of one notated chunk (adjusted notation record)."""
    hz, mask = np.zeros(n), np.zeros(n, bool)
    f = lambda t: min(n, max(0, int(round(t / HOP))))
    mask[:] = True                                   # silent unless inside a stretch
    for s in rec["segments"]:
        mask[f(s["t0"] - SILENT_MARGIN_S):f(s["t1"] + SILENT_MARGIN_S)] = False
    for s in rec["segments"]:
        for nt in s["notes"] or []:
            if nt["t0"] is None:
                continue
            sw, o = raagdb.parse_phrase([nt["swar"]])
            if not sw:
                continue
            hz[f(nt["t0"]):f(nt["t1"])] = tonic * 2 ** ((100 * sw[0] + 1200 * o[0]) / 1200)
            mask[f(nt["t0"]):f(nt["t1"])] = True
    return hz, mask


def melodia(video, t0, n):
    z = fullaudio._cache()
    f0, conf, hop = z[f"{video}|f0"], z[f"{video}|conf"], float(z[f"{video}|hop"])
    a, b = int(round(t0 / hop)), int(round(t0 / hop + n * HOP / hop)) + 1
    return f0[a:b], conf[a:b], hop


def items():
    idx, out = fullaudio.index(), []
    for rec in corpus.adjusted():
        if rec["bad_tonic"]:
            continue
        v, t0, t1 = rec["video"], rec["t0"], rec["t1"]
        n = int(round((t1 - t0) / HOP))
        hz, mask = labels(rec, idx[v].tonic_hz, n)
        out.append(dict(chunk=rec["chunk_id"], video=v, t0=t0, t1=t1, n=n, hz=hz, mask=mask,
                        path=idx[v].path))
    return out


def blocks(it, config):
    out = []
    for name in config.split("+"):
        if name == "melodia":
            out.append(features.track_block(*melodia(it["video"], it["t0"], it["n"]), it["n"]))
        else:
            out.append(features.extract(name, it["path"], it["t0"], it["t1"]))
    n = min(len(b) for b in out)
    return [b[:n] for b in out]


def baseline(it, name):
    """The track a plain source gives on this chunk's grid."""
    if name == "melodia":
        f0, _, hop = melodia(it["video"], it["t0"], it["n"])
        return features.on_grid(f0, hop, it["n"])
    z = fullaudio._cache()
    hop = float(z[f"{it['video']}|hop"])
    f0 = source.on_grid(it["video"], len(z[f"{it['video']}|f0"]), hop, name)
    return features.on_grid(f0[int(round(it["t0"] / hop)):], hop, it["n"])


def scores(pairs):
    """Frame metrics pooled over (predicted Hz, item) pairs; see ../transcriber/plan.md."""
    hit = chroma = vrec = note = fa = sil = 0
    for pred, it in pairs:
        n = min(len(pred), it["n"])
        p, hz, m = pred[:n], it["hz"][:n], it["mask"][:n]
        on, off = m & (hz > 0), m & (hz == 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            d = np.abs(1200 * np.log2(p[on] / hz[on]))
        ok = p[on] > 0
        hit += np.sum(ok & (d <= AGREE_CENTS))
        chroma += np.sum(ok & (np.abs((d + 600) % 1200 - 600) <= AGREE_CENTS))
        vrec += ok.sum(); note += on.sum(); fa += np.sum(p[off] > 0); sil += off.sum()
    return dict(pitch_acc=hit / note, pitch_class_acc=chroma / note, voicing_recall=vrec / note,
                false_voicing=fa / max(sil, 1), note_frames=int(note), silent_frames=int(sil))


def crossval(its, config, log=print):
    st, fold = corpus.folds()
    fold_of = {s["recording"]: f for s, f in zip(st, fold)}
    data = [dict(it, blocks=blocks(it, config)) for it in its]
    for d in data:
        d["hz"], d["mask"] = d["hz"][:len(d["blocks"][0])], d["mask"][:len(d["blocks"][0])]
    pairs = []
    for f in sorted(set(fold_of.values())):
        log(f"  {config}: fold {f + 1}")
        head = adapt.fit([d for d in data if fold_of[d["video"]] != f], log=log)
        pairs += [(adapt.predict(head, d["blocks"]).f0_hz, d) for d in data if fold_of[d["video"]] == f]
    return scores(pairs)


def main(configs):
    import torch
    torch.set_num_threads(4)
    its = items()
    print(f"{len(its)} notated chunks, {len({i['video'] for i in its})} recordings")
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for name in ("melodia", "basic_pitch"):
        res[f"{name} (no head)"] = scores([(baseline(it, name), it) for it in its])
    for cfg in configs:
        res[f"head on {cfg}"] = crossval(its, cfg)
        OUT.parent.mkdir(exist_ok=True)
        OUT.write_text(json.dumps(res, indent=1, default=float))
    print(f"\n{'':28s} {'pitch':>6s} {'class':>6s} {'voiced':>7s} {'false v.':>8s}")
    for k, r in res.items():
        print(f"{k:28s} {r['pitch_acc']:6.3f} {r['pitch_class_acc']:6.3f} {r['voicing_recall']:7.3f} "
              f"{r['false_voicing']:8.3f}")


if __name__ == "__main__":
    main(sys.argv[1:])
