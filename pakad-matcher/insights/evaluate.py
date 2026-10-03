"""Score the insight functions against Neeraja's insight-clip annotations.

    poetry run python -m insights.evaluate --val     # choose on train + validation only, freeze
    poetry run python -m insights.evaluate --test    # score the frozen choice on test (once)

Directions: per labelled swar (aarohi / avarohi / both; "not sung" and "unsure" are skipped) the
machine says aarohi, avarohi or neither (= both). Score: balanced accuracy, the mean recall over
the three classes -- "both" is most swars, so plain accuracy would reward saying nothing.

Nyas: each machine pause event (swar, time) is matched one-to-one to a marked nyas window if the
pause starts within [window start - EARLY_S, window end + LATE_S]. Score: F1 of matched events whose
swar (pitch class) is right. Also reported: detection P/R ignoring the swar, swar accuracy of
matched events, and F1 of the clip-level nyas *set* against the set of swars marked.

Audio only: clips are read with no raag and no scale. Selection (--val) never touches the test
clips: leave-one-clip-out over train + validation (the 7 validation clips alone cannot separate
variants), *every* variant refitted inside each fold -- the threshold heuristics included. The
chosen variant per question is refitted on train + validation and frozen (choice.json + detector
numbers); --test then only loads and scores. Intervals: metrics.bootstrap over clips.
Terms: [DATA.md § Glossary](../DATA.md#glossary).
"""

import argparse
import itertools
import json

import numpy as np

import _bootstrap  # noqa: F401
import audit
import config as C
import fit_reader
import fullaudio
import metrics
import notes
from insights import clips, core, detect, voice
from utils import raagdb

EARLY_S, LATE_S = 0.3, 0.6
DIR_GRID = dict(dir_ratio=(2.0, 3.0, 5.0, 10.0), dir_min_count=(2, 3, 4))
PAUSE_GRID = dict(pause_min_s=(0.06, 0.1, 0.15), pause_rel=(0.5, 1.0, 2.0, 4.0),
                  pause_abs_s=(0.25, 0.4, 0.6))
LEARNED = {"learned": False, "learned + notation": True}   # name: notation as direction labels
NAMES = raagdb.SWAR_NAMES


def load(split):
    """Labelled clips of a split, each read once: contour, reading, loudness, labels."""
    labels, reader, out = clips.labels(), fit_reader.load(), []
    for ch in clips.registry():
        lab = labels.get(ch["id"])
        if ch["split"] != split or not lab or not (lab["directions"] or lab["nyas"]) \
                or ch["video"] in C.BAD_TONIC_VIDEOS:
            continue                          # unlabelled, saved empty (skipped), or wrong tonic
        ctr = fullaudio.contour(ch["video"])
        a, b = int(round(ch["t0"] / ctr.hop)), int(round(ch["t1"] / ctr.hop))
        cents, wav = ctr.cents[a:b], C.INSIGHT_CLIP_DIR / f"{ch['id']}.wav"
        out.append(dict(clip=ch, lab=lab, cents=cents, hop=ctr.hop,
                        read=notes.read(cents, ctr.hop, reader),
                        loud=core.loudness(wav, ctr.hop, len(cents)),
                        voice=voice.voice_db(wav, ctr.hop, len(cents))))
    return out


def pc(token):
    return raagdb.parse_phrase([token])[0][0] % 12


def dir_score(items, predict):
    """`predict(item)` -> {swar: 'aarohi' | 'avarohi'}; anything else counts as 'both'."""
    truth, pred = [], []
    for it in items:
        said = {NAMES[s]: k for s, k in predict(it).items()}
        for sw, v in it["lab"]["directions"].items():
            if v in ("aarohi", "avarohi", "both"):
                truth.append(v); pred.append(said.get(sw, "both"))
    truth, pred = np.array(truth), np.array(pred)
    rec = {k: float(np.mean(pred[truth == k] == k)) for k in ("aarohi", "avarohi", "both")
           if (truth == k).any()}
    prec = {k: float(np.mean(truth[pred == k] == k)) for k in ("aarohi", "avarohi")
            if (pred == k).any()}
    return dict(balanced=float(np.mean(list(rec.values()))) if rec else np.nan, recall=rec,
                precision=prec, n={k: int((truth == k).sum()) for k in ("aarohi", "avarohi", "both")})


def nyas_score(items, events):
    """`events(item)` -> [(swar, time s)] of predicted nyas."""
    tp = tp_any = n_pred = n_true = 0
    set_tp = set_pred = set_true = 0
    for it in items:
        ev = events(it)
        wins = it["lab"]["nyas"]
        n_pred += len(ev); n_true += len(wins)
        used = set()
        for s, t in ev:
            cand = [i for i, w in enumerate(wins) if i not in used
                    and w["t0"] - EARLY_S <= t <= w["t1"] + LATE_S]
            if not cand:
                continue
            i = min(cand, key=lambda i: abs(wins[i]["t1"] - t))
            used.add(i); tp_any += 1; tp += pc(wins[i]["swar"]) == s
        mine = set(core.nyas([s for s, _ in ev])["nyas"])
        hers = {pc(w["swar"]) for w in wins}
        set_tp += len(mine & hers); set_pred += len(mine); set_true += len(hers)
    f1 = lambda a, b, c: 2 * a / max(b + c, 1)
    return dict(f1=f1(tp, n_pred, n_true), detect_precision=tp_any / max(n_pred, 1),
                detect_recall=tp_any / max(n_true, 1), swar_right=tp / max(tp_any, 1),
                set_f1=f1(set_tp, set_pred, set_true), events=n_pred, marked=n_true)


def nyas_f1_events(items, events):
    return nyas_score(items, events)["f1"]


def tune_heuristics(items):
    """Threshold heuristics by grid search on `items`: directions, then pauses."""
    best = dict(C.INSIGHTS)
    for grid, score, key, make in ((DIR_GRID, dir_score, "balanced", detect.rule_direction),
                                   (PAUSE_GRID, nyas_score, "f1", detect.rule_nyas)):
        top, top_s = best, -1
        for vals in itertools.product(*grid.values()):
            p = dict(best, **dict(zip(grid, vals)))
            sc = score(items, make(p))[key]
            if sc > top_s + 1e-9:
                top, top_s = p, sc
        best = top
    return best


def fit_variant(name, items, notation, leave_out_video=None):
    """(direction predictor, nyas predictor, what to freeze) for one variant, fitted on `items`."""
    if name == "threshold heuristics":
        p = tune_heuristics(items)
        return detect.rule_direction(p), detect.rule_nyas(p), dict(rules=p)
    extra = [n for n in notation if n["clip"]["video"] != leave_out_video] if LEARNED[name] else ()
    m = detect.fit(items, extra)
    return detect.learned_direction(m), detect.learned_nyas(m), dict(models=m)


def show(name, items, dpred, npred):
    d, n = dir_score(items, dpred), nyas_score(items, npred)
    print(f"  {name:22s} directions {d['balanced']:.3f} (recall aar {d['recall'].get('aarohi', 0):.2f} "
          f"ava {d['recall'].get('avarohi', 0):.2f} both {d['recall'].get('both', 0):.2f})   "
          f"nyas F1 {n['f1']:.3f} (detect P {n['detect_precision']:.2f} R {n['detect_recall']:.2f}, "
          f"swar right {n['swar_right']:.2f}, set F1 {n['set_f1']:.2f}, {n['events']} ev / {n['marked']})")
    return dict(directions=d, nyas=n)


def validate():
    items = load("train") + load("validation")
    notation = detect.notation_items()
    names = ["threshold heuristics"] + list(LEARNED)
    print(f"{len(items)} clips (train + validation), leave-one-clip-out, every variant refitted "
          f"per fold; {sum(len(n['lab']['directions']) for n in notation)} notation proxy labels")
    out = {k: ({}, {}) for k in names}
    for i, it in enumerate(items):
        rest = items[:i] + items[i + 1:]
        for name in names:
            dp, npd, _ = fit_variant(name, rest, notation, it["clip"]["video"])
            out[name][0][id(it)], out[name][1][id(it)] = dp(it), npd(it)
    res = {name: show(name, items, lambda it, o=o: o[0][id(it)], lambda it, o=o: o[1][id(it)])
           for name, o in out.items()}
    picks = {"directions": max(names, key=lambda k: res[k]["directions"]["balanced"]),
             "nyas": max(names, key=lambda k: res[k]["nyas"]["f1"])}
    print(f"\nchosen -- directions: {picks['directions']};  nyas: {picks['nyas']}")
    frozen = {}
    heur = tune_heuristics(items)                  # always frozen: the no-audio fallback
    for task, pick in picks.items():
        if pick == "threshold heuristics":
            frozen[task] = dict(choice=pick, rules=heur)
        else:
            path = C.INSIGHTS_DIR / f"detectors_{task}.json"
            detect.save(fit_variant(pick, items, notation)[2]["models"], path)
            frozen[task] = dict(choice=pick, detectors=path.name)
    keys = list(DIR_GRID) + list(PAUSE_GRID)
    print("  threshold heuristics on all train + validation: "
          + ", ".join(f"{k}={heur[k]}" for k in keys))
    detect.CHOICE_JSON.write_text(json.dumps(dict(
        **frozen, heuristics=heur, splits_manifest=audit.manifest_hash(),
        cv={k: dict(directions=v["directions"]["balanced"], nyas_f1=v["nyas"]["f1"])
            for k, v in res.items()}), indent=1, default=float))
    print(f"-> {detect.CHOICE_JSON}")


def _diff_ci(items, score):
    """Paired difference over test clips, with a 95% interval from resampling clips."""
    return metrics.bootstrap(lambda g: score([x for c in g for x in c]), [[it] for it in items])


def test():
    """Load the frozen predictors and score the test clips. No fitting happens here."""
    choice = json.loads(detect.CHOICE_JSON.read_text())
    if choice.get("splits_manifest") != audit.manifest_hash():
        print("WARNING: the splits changed since the choice was frozen (audit.py --freeze)")
    items = load("test")
    dp, npd = detect.frozen("directions"), detect.frozen("nyas")
    hd, hn = detect.rule_direction(choice["heuristics"]), detect.rule_nyas(choice["heuristics"])
    print(f"test: {len(items)} clips; frozen: directions '{dp.method}', nyas '{npd.method}'")
    res = dict(chosen=show("chosen (frozen)", items, dp, npd),
               heuristics=show("threshold heuristics", items, hd, hn))
    d = _diff_ci(items, lambda its: dir_score(its, dp)["balanced"] - dir_score(its, hd)["balanced"])
    n = _diff_ci(items, lambda its: nyas_score(its, npd)["f1"] - nyas_score(its, hn)["f1"])
    for name, (est, lo, hi) in (("directions", d), ("nyas F1", n)):
        print(f"  chosen - heuristics, {name}: {est:+.3f}  95% interval [{lo:+.3f}, {hi:+.3f}]")
    res.update(difference_ci=dict(directions=d, nyas_f1=n), splits_manifest=audit.manifest_hash())
    (C.INSIGHTS_DIR / "test.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", action="store_true")
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    validate() if a.val else test() if a.test else ap.print_help()
