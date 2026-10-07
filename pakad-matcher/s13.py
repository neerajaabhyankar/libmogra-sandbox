"""S13: the phrase matcher run on the reader's notes instead of raw Melodia. Validation only.

    poetry run python s13.py

The reader (notation-fitted, frozen in results/reader.json) reads each judged span; its notes are
written back as a cleaned pitch track -- each note's frames set to its swar's pitch -- and the
usual matcher scores the samooha on that track. Two ways to build the track:

  notes only          frames outside the reader's notes are unvoiced
  notes + transitions frames outside the reader's notes keep their raw pitch (meend, kan)

Each is scored with the hand-set costs and val-tuned (leave-one-samooha-out, as in s7). Nothing
here touches test: if a variant beats the frozen val-tuned on validation, it goes into s7.py and
test is scored once (plan.md, "To try"). Terms: [DATA.md § Glossary](DATA.md#glossary).
"""

import json

import numpy as np

import _bootstrap  # noqa: F401
import config as C
import corpus
import fit_reader
import metrics
import notes
import s7

VARIANTS = {"notes only": False, "notes + transitions": True}
OUT = C.RESULTS_DIR / "s13_val.json"


def cleaned(span, transitions, reader):
    """`span` with its contour replaced by the reader's notes, each held at its swar's pitch."""
    cents, hop = np.asarray(span["cents"], float), span["hop"]
    out = cents.copy() if transitions else np.full_like(cents, np.nan)
    for _, _, ns in notes.read(cents, hop, reader):
        for _, pitch, t0, t1 in ns:
            out[int(round(t0 / hop)):int(round(t1 / hop))] = pitch
    return dict(span, cents=out)


def loso_tuned(spans):
    pid = np.array([s["pid"] for s in spans])
    out = np.zeros(len(spans))
    for p in sorted(set(pid)):
        params = s7.val_tuned([s for s, q in zip(spans, pid) if q != p])
        out[pid == p] = -s7.match_cost([s for s, q in zip(spans, pid) if q == p], params)
    return out


def main():
    spans, reader = corpus.spans("validation"), fit_reader.load()
    frozen = json.loads(s7.CHOICE_JSON.read_text())
    print(f"VALIDATION: {len(spans)} spans, {len({s['pid'] for s in spans})} samoohas; "
          f"frozen choice {frozen['chosen']} at AUC {frozen['val_auc']:.3f}\n")
    sc = {"raw Melodia, hand-set": -s7.match_cost(spans, s7.HAND_SET)}
    for name, keep in VARIANTS.items():
        cl = [cleaned(s, keep, reader) for s in spans]
        sc[f"reader {name}, hand-set"] = -s7.match_cost(cl, s7.HAND_SET)
        sc[f"reader {name}, val-tuned (leave-one-samooha-out)"] = loso_tuned(cl)
        print(f"  {name} done", flush=True)
    s7.table(spans, sc)
    OUT.write_text(json.dumps({k: dict(auc=metrics.per_samooha_auc(v, spans),
                                       p1=metrics.precision_at(v, spans, 1),
                                       p3=metrics.precision_at(v, spans, 3)) for k, v in sc.items()}
                              | dict(frozen_val_auc=frozen["val_auc"]), indent=1))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
