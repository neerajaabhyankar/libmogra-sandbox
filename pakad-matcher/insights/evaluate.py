"""Score the insight functions against Neeraja's insight-clip annotations.

    poetry run python -m insights.evaluate --val     # tune on train, choose on validation, freeze
    poetry run python -m insights.evaluate --test    # score test once with the frozen choice

Directions: per labelled swar (aarohi / avarohi / both; "not sung" and "unsure" are skipped) the
machine says aarohi, avarohi or neither (= both). Score: balanced accuracy, the mean recall over
the three classes -- "both" is most swars, so plain accuracy would reward saying nothing.

Nyas: each machine pause event (swar, time) is matched one-to-one to a marked nyas window if the
pause starts within [window start - EARLY_S, window end + LATE_S]. Score: F1 of matched events whose
swar (pitch class) is right. Also reported: detection P/R ignoring the swar, swar accuracy of
matched events, and F1 of the clip-level nyas *set* against the set of swars marked.

**Audio only**: clips are read with no raag and no scale, as unknown audio would be.
Variants: the threshold heuristics as configured (I3, tuned on train), and the learned detectors
(insights/detect.py), with and without notation as extra direction labels. Choice (--val): leave-one-clip-out over train +
validation together -- the 7-clip validation set alone (3 aarohi labels) cannot tell variants apart.
Directions and nyas are chosen separately, the chosen learned models are refitted on train +
validation and frozen as numbers (results/insights/detectors_*.json); --test scores test once
with exactly those numbers.
Terms: [DATA.md § Glossary](../DATA.md#glossary).
"""

import argparse
import itertools
import json

import numpy as np

import _bootstrap  # noqa: F401
import config as C
import fit_reader
import fullaudio
from insights import clips, core, detect, voice
from utils import raagdb

EARLY_S, LATE_S = 0.3, 0.6
DIR_GRID = dict(dir_ratio=(2.0, 3.0, 5.0, 10.0), dir_min_count=(2, 3, 4),
                dir_min_note_s=(0.08, 0.12, 0.16))
PAUSE_GRID = dict(pause_min_s=(0.06, 0.1, 0.15, 0.25), pause_rel=(0.0, 0.5, 1.0, 2.0),
                  skip_short_s=(0.0, 0.1))
DROP_GRID = (3.0, 6.0, 10.0)
CHOICE = C.INSIGHTS_DIR / "choice.json"
NAMES = raagdb.SWAR_NAMES


def load(split):
    """Labelled clips of a split, each read once: contour, reading, loudness, labels."""
    labels, reader, out = clips.labels(), fit_reader.load(), []
    for ch in clips.registry():
        lab = labels.get(ch["id"])
        if ch["split"] != split or not lab or not (lab["directions"] or lab["nyas"]):
            continue                                     # unlabelled, or saved empty (skipped)
        ctr = fullaudio.contour(ch["video"])
        a, b = int(round(ch["t0"] / ctr.hop)), int(round(ch["t1"] / ctr.hop))
        cents = ctr.cents[a:b]
        out.append(dict(clip=ch, lab=lab, cents=cents, hop=ctr.hop,
                        read=core.read(cents, ctr.hop, reader),           # audio only: no raag
                        loud=core.loudness(C.INSIGHT_CLIP_DIR / f"{ch['id']}.wav", ctr.hop, len(cents)),
                        voice=voice.voice_db(C.INSIGHT_CLIP_DIR / f"{ch['id']}.wav", ctr.hop, len(cents))))
    return out


def pc(token):
    return raagdb.parse_phrase([token])[0][0] % 12


def rule_dir(p):
    def f(it):
        uni = core.unidirectional(core.moves(it["read"], p), p["dir_ratio"], p["dir_min_count"])
        return {s: k for k in ("aarohi", "avarohi") for s in uni[k]}
    return f


def rule_nyas(p):
    return lambda it: core.nyas_events(it["cents"], it["hop"], it["read"], p, it["loud"])


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
    return dict(balanced=float(np.mean(list(rec.values()))), recall=rec, precision=prec,
                n={k: int((truth == k).sum()) for k in ("aarohi", "avarohi", "both")})


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


def tune(items, grid, score, key, base, make):
    best, best_s = None, -1
    for vals in itertools.product(*grid.values()):
        p = dict(base, **dict(zip(grid, vals)))
        sc = score(items, make(p))[key]
        if sc > best_s + 1e-9:
            best, best_s = p, sc
    return best


LEARNED = {                      # name: (notation as direction labels, nyas L2 C)
    "learned": (False, None),
    "learned + notation": (True, None),
    "learned, nyas C 0.1": (False, 0.1),
    "learned, nyas C 2": (False, 2.0),
}


def fit_variant(spec, items, notation, leave_out_video=None):
    use_nota, c = spec
    extra = [n for n in notation if n["clip"]["video"] != leave_out_video] if use_nota else ()
    return detect.fit(items, extra, c)


def predictors(m):
    return (lambda it: detect.direction_predict(it, m["direction"]),
            lambda it: detect.nyas_predict(it, m["nyas"], m["nyas_threshold"]))


def show(name, items, dpred, npred):
    d, n = dir_score(items, dpred), nyas_score(items, npred)
    print(f"  {name:28s} directions {d['balanced']:.3f} (recall aar {d['recall'].get('aarohi', 0):.2f} "
          f"ava {d['recall'].get('avarohi', 0):.2f} both {d['recall'].get('both', 0):.2f})   "
          f"nyas F1 {n['f1']:.3f} (detect P {n['detect_precision']:.2f} R {n['detect_recall']:.2f}, "
          f"swar right {n['swar_right']:.2f}, set F1 {n['set_f1']:.2f}, {n['events']} ev / {n['marked']})")
    return dict(directions=d, nyas=n)


def validate():
    items = load("train") + load("validation")
    notation = detect.notation_items()
    print(f"{len(items)} clips (train + validation), leave-one-clip-out; "
          f"{sum(len(n['lab']['directions']) for n in notation)} notation proxy direction labels "
          f"(left out for the held-out clip's recording)")
    train = [it for it in items if it["clip"]["split"] == "train"]
    heur = tune(train, DIR_GRID, dir_score, "balanced", dict(C.INSIGHTS), rule_dir)
    heur = tune(train, PAUSE_GRID, nyas_score, "f1", heur, rule_nyas)       # audio only, on train
    held = {"threshold heuristics": predictors_rules(heur)}
    held.update({k: ({}, {}) for k in LEARNED})
    out = {k: ({}, {}) for k in held}
    for i, it in enumerate(items):
        dp, npd = held["threshold heuristics"]
        out["threshold heuristics"][0][i], out["threshold heuristics"][1][i] = dp(it), npd(it)
        rest = items[:i] + items[i + 1:]
        for name, spec in LEARNED.items():
            dp, npd = predictors(fit_variant(spec, rest, notation, it["clip"]["video"]))
            out[name][0][i], out[name][1][i] = dp(it), npd(it)
    idx = {id(it): i for i, it in enumerate(items)}
    res = {name: show(name, items, lambda it, o=o: o[0][idx[id(it)]], lambda it, o=o: o[1][idx[id(it)]])
           for name, o in out.items()}
    pick_d = max(res, key=lambda k: res[k]["directions"]["balanced"])
    pick_n = max(res, key=lambda k: res[k]["nyas"]["f1"])
    print(f"\nchosen -- directions: {pick_d};  nyas: {pick_n}")
    frozen = {}
    for task, pick in (("directions", pick_d), ("nyas", pick_n)):
        if pick in LEARNED:
            m = fit_variant(LEARNED[pick], items, notation)
            path = C.INSIGHTS_DIR / f"detectors_{task}.json"
            detect.save(m, path)
            frozen[task] = dict(choice=pick, detectors=path.name)
        else:
            frozen[task] = dict(choice=pick, rules=heur)
    keys = list(DIR_GRID) + list(PAUSE_GRID)
    print("  threshold heuristics retuned on train (audio only): "
          + ", ".join(f"{k}={heur[k]}" for k in keys))
    CHOICE.write_text(json.dumps(dict(**frozen, heuristics=heur, cv={k: dict(directions=v["directions"]["balanced"],
                                                            nyas_f1=v["nyas"]["f1"])
                                                     for k, v in res.items()}), indent=1, default=float))
    print(f"-> {CHOICE}")


def predictors_rules(p):
    return rule_dir(p), rule_nyas(p)


def frozen_predictor(task, entry):
    if "rules" in entry:
        return predictors_rules(entry["rules"])[0 if task == "directions" else 1]
    return predictors(detect.load(C.INSIGHTS_DIR / entry["detectors"]))[0 if task == "directions" else 1]


def test():
    ch = json.loads(CHOICE.read_text())
    items = load("test")
    print(f"test: {len(items)} clips; frozen: directions '{ch['directions']['choice']}', "
          f"nyas '{ch['nyas']['choice']}'")
    res = dict(baseline=show("threshold heuristics", items, *predictors_rules(ch["heuristics"])))
    res["chosen"] = show("chosen (frozen)", items, frozen_predictor("directions", ch["directions"]),
                         frozen_predictor("nyas", ch["nyas"]))
    (C.INSIGHTS_DIR / "test.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", action="store_true")
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    validate() if a.val else test() if a.test else ap.print_help()
