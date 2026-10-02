"""Learned insight detectors: small logistic models on cues from the tuned heuristic notes.

**Audio only.** Nothing here takes a raag: at use time a clip comes with its Sa, never its raag.
Raag labels are used only to *train* (the notation's raag-specific readings are not needed either).

  nyas       every note end is a candidate "a breath/pause follows here". Cues: the pitch-track gap
             after it (and that gap over the local pace), loudness and voice-above-drone loudness
             falling from the note to just after it, the note's length, whether it ends a stretch
             between silences, the pitch falling into its end, Sa, Pa. Positive = the candidate
             nearest the end of a marked nyas window. Events closer than nms_s keep the likelier.
  direction  per swar of a clip: aarohi / avarohi / both, from up/down counts at several minimum
             note lengths, a smoothed up-fraction, and the time spent on the swar.

Models are plain logistic regressions, written out as numbers so they can be ported.
Terms: [DATA.md § Glossary](../DATA.md#glossary).
"""

import json

import numpy as np
from sklearn.linear_model import LogisticRegression

import config as C
from insights import core
from utils import raagdb

D = C.INSIGHT_DETECT
DIR_CLASSES = ("aarohi", "avarohi", "both")


# ---------------------------------------------------------------- nyas
def _lvl(x, a, b):
    a, b = max(0, a), min(len(x), b)
    return float(np.mean(x[a:b])) if b > a else 0.0


def nyas_candidates(cents, hop, read_, loud, voice):
    """[(swar, end time s, feature vector)] for every note end."""
    cents = np.asarray(cents, float)
    voiced = ~np.isnan(cents)
    ns_all = [(n, k == len(ns) - 1) for _, _, ns in read_ for k, n in enumerate(ns)]
    out = []
    for n, last in ns_all:
        s, _c, t0, t1 = n
        e = int(round(t1 / hop))
        u = e
        while u < len(cents) and not voiced[u]:
            u += 1
        if u >= len(cents):
            continue                                    # the clip is cut here: no evidence after
        gap = (u - e) * hop
        near = [m[3] - m[2] for m, _ in ns_all if abs((m[2] + m[3]) / 2 - t1) <= D["pace_window_s"]]
        pace = float(np.median(near)) if near else 0.2
        a0 = int(round(t0 / hop))
        after = slice(max(e - int(0.2 / hop), a0), min(len(cents), e + int(0.8 / hop)))
        tail = cents[max(a0, e - int(0.2 / hop)):e]
        tail = tail[~np.isnan(tail)]
        slope = (tail[-1] - tail[0]) / max(len(tail) * hop, hop) / 1000.0 if len(tail) > 2 else 0.0
        x = [np.log1p(gap / 0.1), np.log1p(gap / max(pace, 0.05)),
             (_lvl(voice, a0, e) - float(np.min(voice[after]))) / 10.0,
             (_lvl(loud, a0, e) - float(np.min(loud[after]))) / 10.0,
             np.log(max(t1 - t0, 0.02) / 0.2), float(last), float(np.clip(slope, -3, 3)),
             float(s == 0), float(s == 7)]
        out.append((s, round(t1, 2), np.array(x)))
    return out


def label_candidates(cands, windows):
    """1 for the candidate nearest each window's end (within the match tolerance), else 0."""
    from insights.evaluate import EARLY_S, LATE_S
    y = np.zeros(len(cands), int)
    for w in windows:
        near = [i for i, (_, t, _) in enumerate(cands) if w["t0"] - EARLY_S <= t <= w["t1"] + LATE_S]
        if near:
            y[min(near, key=lambda i: abs(cands[i][1] - w["t1"]))] = 1
    return y


def nms(events):
    """events [(swar, t, p)] -> keep the likelier of any two closer than nms_s."""
    keep = []
    for e in sorted(events, key=lambda e: -e[2]):
        if all(abs(e[1] - k[1]) >= D["nms_s"] for k in keep):
            keep.append(e)
    return sorted(keep, key=lambda e: e[1])


# ---------------------------------------------------------------- direction
def dir_features(read_):
    """{swar: feature vector} for every swar with at least one move."""
    per = [core.moves(read_, dict(dir_min_note_s=m)) for m in D["dir_min_notes"]]
    dwell = {}
    for _, _, ns in read_:
        for s, _, t0, t1 in ns:
            dwell[s] = dwell.get(s, 0.0) + (t1 - t0)
    out = {}
    for s in set().union(*per):
        x = []
        for m in per:
            u, d = m.get(s, [0, 0])
            x += [(u + 1) / (u + d + 2) - 0.5, np.log1p(u + d)]
        x.append(np.log1p(dwell.get(s, 0.0)))
        out[s] = np.array(x)
    return out


# ---------------------------------------------------------------- notation as direction labels
def notation_items(reader=None):
    """Notated chunks as extra direction training: per swar, the notation's own up/down counts
    give a proxy label -- aarohi if >= proxy_one_way of >= proxy_min moves go up, avarohi the
    mirror, both if within proxy_both; ambiguous swars are left out. Machine features come from
    reading the chunk's contour exactly as for a clip (no raag)."""
    import audit
    import fit_reader
    import fullaudio
    reader = reader or fit_reader.load()
    ch, out = audit.chunks(), []
    for cid, rec in sorted(audit.notations().items()):
        c = ch[cid]
        if c["video"] in C.BAD_TONIC_VIDEOS:
            continue
        counts = {}
        for seg in rec["segments"]:
            sw, oc = raagdb.parse_phrase(seg["swars"].split())
            core.directions([(x % 12, 100 * (x % 12) + 1200 * o, 0, 0) for x, o in zip(sw, oc)], counts)
        labels = {}
        for x, (u, d) in counts.items():
            if u + d < D["proxy_min"]:
                continue
            f = u / (u + d)
            y = ("aarohi" if f >= D["proxy_one_way"] else "avarohi" if f <= 1 - D["proxy_one_way"]
                 else "both" if D["proxy_both"][0] <= f <= D["proxy_both"][1] else None)
            if y:
                labels[raagdb.SWAR_NAMES[x]] = y
        if not labels:
            continue
        ctr = fullaudio.contour(c["video"])
        a, b = int(round(c["t0"] / ctr.hop)), int(round(c["t1"] / ctr.hop))
        out.append(dict(clip=dict(video=c["video"], id=cid),
                        read=core.read(ctr.cents[a:b], ctr.hop, reader),
                        lab=dict(directions=labels, nyas=[])))
    return out


# ---------------------------------------------------------------- fit / predict / save / load
def _model(c=None):
    return LogisticRegression(C=c or D["l2_C"], class_weight="balanced", max_iter=2000)


def fit(items, extra_dir=(), nyas_C=None):
    """Both models from labelled, already-read clips (insights.evaluate.load); `extra_dir` adds
    direction-only items (notation_items)."""
    X, y = [], []
    for it in items:
        cands = nyas_candidates(it["cents"], it["hop"], it["read"], it["loud"], it["voice"])
        if cands:
            X += [c[2] for c in cands]; y += list(label_candidates(cands, it["lab"]["nyas"]))
    ny = _model(nyas_C).fit(np.array(X), np.array(y))
    Xd, yd = [], []
    for it in list(items) + list(extra_dir):
        f = dir_features(it["read"])
        width = len(next(iter(f.values()))) if f else 0
        for sw, v in it["lab"]["directions"].items():
            if v in DIR_CLASSES and width:
                Xd.append(f.get(raagdb.SWAR_NAMES.index(sw), np.zeros(width))); yd.append(v)
    dr = _model().fit(np.array(Xd), np.array(yd))
    return dict(nyas=ny, nyas_threshold=_best_threshold(items, ny), direction=dr)


def _best_threshold(items, ny):
    from insights.evaluate import nyas_f1_events
    best, best_f = 0.5, -1
    for thr in D["thresholds"]:
        f = nyas_f1_events(items, lambda it: nyas_predict(it, ny, thr))
        if f > best_f + 1e-9:
            best, best_f = thr, f
    return best


def nyas_predict(it, ny, thr):
    cands = nyas_candidates(it["cents"], it["hop"], it["read"], it["loud"], it["voice"])
    if not cands:
        return []
    p = ny.predict_proba(np.array([c[2] for c in cands]))[:, 1]
    return [(s, t) for s, t, _ in nms([(c[0], c[1], q) for c, q in zip(cands, p) if q >= thr])]


def direction_predict(it, dr):
    """{swar: 'aarohi' | 'avarohi'} -- swars predicted 'both' are left out."""
    f = dir_features(it["read"])
    if not f:
        return {}
    sw = sorted(f)
    return {s: k for s, k in zip(sw, dr.predict(np.array([f[s] for s in sw]))) if k != "both"}


def save(models, path):
    """Coefficients as plain numbers (portable)."""
    lr = lambda m: dict(classes=[str(c) for c in m.classes_], coef=m.coef_.round(4).tolist(),
                        intercept=m.intercept_.round(4).tolist())
    path.write_text(json.dumps(dict(nyas=lr(models["nyas"]), nyas_threshold=models["nyas_threshold"],
                                    direction=lr(models["direction"]), settings=D), indent=1))


def load(path):
    """Frozen detectors, rebuilt from their numbers."""
    d = json.loads(path.read_text())

    def lr(e):
        m = LogisticRegression()
        m.classes_ = np.array([int(c) for c in e["classes"]] if e["classes"][0] in ("0", "1")
                              else e["classes"])
        m.coef_, m.intercept_ = np.array(e["coef"]), np.array(e["intercept"])
        return m
    return dict(nyas=lr(d["nyas"]), nyas_threshold=d["nyas_threshold"], direction=lr(d["direction"]))


def load_choice(task):
    """Frozen detectors for 'directions' or 'nyas', as chosen by insights/evaluate.py --val."""
    choice = json.loads((C.INSIGHTS_DIR / "choice.json").read_text())[task]
    return load(C.INSIGHTS_DIR / choice["detectors"])
