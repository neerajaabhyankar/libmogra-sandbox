"""S7b: rank the judged spans with models fitted without the test, choose on validation, test once.

    poetry run python s7.py --val      # score every method on validation, record the choice
    poetry run python s7.py --test     # score on test, once, with the choice already frozen

The methods, and what each was allowed to see:

  hand-set        the matcher's costs before S4b -- hand-chosen, never saw a judgment.  BASELINE
  S4b-tuned       config.MATCH as it stands: fitted on the old 168 judgments, which overlap both
                  validation and test. Reported for reference only; never selectable.
  notation-set    hand-set, but tolerance and minimum note length taken from the notation corpus
  read-then-match the notation-fitted reader transcribes the span; the score is how closely the
                  samooha matches some stretch of that transcription (edit distance)
  combined        notation-set cost + a weight on read-then-match; the weight chosen on validation
  val-tuned       hand-set costs re-tuned by coordinate ascent on the validation judgments -- the
                  legitimate version of what S4b did

Selection: highest per-samooha AUC on validation, ties broken by P@3 -- but for any method that
was *fitted* on validation (the combined weight, val-tuned), the number compared is a
leave-one-samooha-out estimate: fit on 11 samoohas, score the 12th. Otherwise the tuned methods
would be graded on the answers they were tuned to, and would always win. (First version of this
file compared in-sample numbers; caught and fixed before the test was touched.)

Terms: [DATA.md § Glossary](DATA.md#glossary).
"""

import argparse
import json

import numpy as np

import _bootstrap  # noqa: F401
import config as C
import corpus
import decode
import fit_reader
import matcher
from contour import Contour

HAND_SET = dict(C.MATCH, free_cents=30.0, note_cap=3.0, held_slope=400.0, note_trim=0.5)
CHOICE_JSON = C.RESULTS_DIR / "s7_choice.json"
TUNE_GRID = {"free_cents": [15.0, 23.0, 30.0, 45.0], "note_trim": [0.5, 0.75, 1.0],
             "held_slope": [400.0, 800.0], "note_cap": [2.0, 3.0],
             "register_penalty": [0.0, 0.5, 1.0]}


def notation_params():
    """Hand-set, with the two quantities the notation corpus measures directly."""
    reader = json.loads(fit_reader.READER_JSON.read_text())
    return dict(HAND_SET, free_cents=23.0, min_dwell_s=0.05, swar_offsets=reader["swar_offsets"])


def match_cost(spans, params):
    out = []
    for s in spans:
        c = matcher.match(Contour("span", s["cents"], s["hop"]), s["swars"], top_k=1,
                          params=params, octaves=s["octaves"])
        out.append(c[0].cost if c else 9.0)
    return np.array(out)


def semiglobal(pattern, text):
    """Edit distance from `pattern` to its best-matching stretch of `text` (text ends are free)."""
    n, m = len(pattern), len(text)
    d = np.zeros((n + 1, m + 1))
    d[:, 0] = np.arange(n + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1,
                          d[i - 1, j - 1] + (pattern[i - 1] != text[j - 1]))
    return d[n].min()


def read_distance(spans):
    reader = json.loads(fit_reader.READER_JSON.read_text())
    params = dict(C.READ_MATCH, swar_offsets=reader["swar_offsets"])
    onsets = (reader["onset_slow"], reader["onset_fast"])
    out = []
    for s in spans:
        seq = fit_reader.read(s, params, onsets)
        want = [x % 12 for x in s["swars"]]
        out.append(semiglobal(want, seq) / len(want))
    return np.array(out)


def val_tuned(spans):
    """Coordinate ascent on validation only -- the legitimate analogue of S4b."""
    best, score = dict(HAND_SET), corpus.per_samooha_auc(-match_cost(spans, HAND_SET), spans)
    for _ in range(2):
        improved = False
        for k, values in TUNE_GRID.items():
            for v in values:
                if best.get(k) == v:
                    continue
                trial = dict(best, **{k: v})
                s = corpus.per_samooha_auc(-match_cost(spans, trial), spans)
                if s > score + 1e-6:
                    best, score, improved = trial, s, True
        if not improved:
            break
    return best


def scores(spans, choice=None):
    """Higher = more likely the samooha, for every method."""
    notation = notation_params()
    out = {
        "hand-set (baseline)": -match_cost(spans, HAND_SET),
        "S4b-tuned (reference: fitted on old judgments)": -match_cost(spans, C.MATCH),
        "notation-set": -match_cost(spans, notation),
        "read-then-match": -read_distance(spans),
    }
    w = (choice or {}).get("combined_weight", 1.0)
    out["combined"] = out["notation-set"] + w * out["read-then-match"]
    if choice and "val_tuned" in choice:
        out["val-tuned"] = -match_cost(spans, choice["val_tuned"])
    return out


def table(spans, sc):
    print(f"{'method':50s} {'AUC':>6s} {'P@1':>6s} {'P@3':>6s}")
    for name, s in sc.items():
        print(f"{name:50s} {corpus.per_samooha_auc(s, spans):6.3f} "
              f"{corpus.precision_at(s, spans, 1):6.2f} {corpus.precision_at(s, spans, 3):6.2f}")


def loso(spans, base):
    """Scores for the validation-fitted methods, each samooha scored by a fit that never saw it."""
    pid = np.array([s["pid"] for s in spans])
    comb, tuned = np.zeros(len(spans)), np.zeros(len(spans))
    ws = (0.25, 0.5, 1.0, 2.0, 4.0)
    for p in sorted(set(pid)):
        tr, te = pid != p, pid == p
        sub = [s for s, m in zip(spans, tr) if m]
        w = ws[int(np.argmax([corpus.per_samooha_auc(base["notation-set"][tr]
                                                     + x * base["read-then-match"][tr], sub) for x in ws]))]
        comb[te] = base["notation-set"][te] + w * base["read-then-match"][te]
        params = val_tuned(sub)
        tuned[te] = -match_cost([s for s, m in zip(spans, te) if m], params)
    return {"combined (leave-one-samooha-out)": comb, "val-tuned (leave-one-samooha-out)": tuned}


def validate():
    spans = corpus.spans("validation")
    print(f"VALIDATION: {len(spans)} spans, {len({s['pid'] for s in spans})} samoohas, "
          f"{np.mean([s['y'] for s in spans]):.0%} yes\n")
    base = scores(spans)
    # the one free weight, chosen here
    ws = (0.25, 0.5, 1.0, 2.0, 4.0)
    auc_w = [corpus.per_samooha_auc(base["notation-set"] + w * base["read-then-match"], spans) for w in ws]
    weight = ws[int(np.argmax(auc_w))]
    tuned = val_tuned(spans)
    choice = dict(combined_weight=weight, val_tuned={k: v for k, v in tuned.items()})
    sc = scores(spans, choice)
    table(spans, sc)

    # honest validation numbers for the two methods fitted on validation
    honest = loso(spans, base)
    print(f"\n{'fitted on validation -> leave-one-samooha-out':50s} {'AUC':>6s} {'P@1':>6s} {'P@3':>6s}")
    for name, s in honest.items():
        print(f"{name:50s} {corpus.per_samooha_auc(s, spans):6.3f} "
              f"{corpus.precision_at(s, spans, 1):6.2f} {corpus.precision_at(s, spans, 3):6.2f}")
    # eligible: methods that never saw validation, plus the *honest* numbers of those that did.
    # The in-sample "combined" and "val-tuned" rows above are shown, never compared.
    eligible = {k: v for k, v in sc.items()
                if not k.startswith("S4b") and k not in ("combined", "val-tuned")}
    eligible.update(honest)
    key = lambda k: (corpus.per_samooha_auc(eligible[k], spans), corpus.precision_at(eligible[k], spans, 3))
    chosen = max(eligible, key=key)
    choice["chosen"] = chosen
    choice["val_auc"] = corpus.per_samooha_auc(eligible[chosen], spans)
    CHOICE_JSON.parent.mkdir(exist_ok=True)
    CHOICE_JSON.write_text(json.dumps(choice, indent=1))
    print(f"\ncombined weight {weight}; val-tuned changes: "
          + ", ".join(f"{k}={v}" for k, v in tuned.items() if HAND_SET.get(k) != v))
    print(f"CHOSEN on validation: {chosen}  -> frozen in {CHOICE_JSON}")


def test():
    choice = json.loads(CHOICE_JSON.read_text())
    spans = corpus.spans("test")
    print(f"TEST: {len(spans)} spans, {len({s['pid'] for s in spans})} samoohas, "
          f"{np.mean([s['y'] for s in spans]):.0%} yes  (choice frozen: {choice['chosen']})\n")
    sc = scores(spans, choice)
    table(spans, sc)
    chosen = choice["chosen"].replace(" (leave-one-samooha-out)", "")
    print(f"\nheadline, chosen on validation before this run: {chosen}")
    json.dump({k: dict(auc=corpus.per_samooha_auc(v, spans), p1=corpus.precision_at(v, spans, 1),
                       p3=corpus.precision_at(v, spans, 3)) for k, v in sc.items()},
              open(C.RESULTS_DIR / "s7_test.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", action="store_true")
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    if a.val:
        validate()
    if a.test:
        test()
