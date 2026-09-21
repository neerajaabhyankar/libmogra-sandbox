"""Phrase catalogue: every mukhyanga phrase of the dataset raags, tiered and ranked.

Tiering (plan.md Q4): drop phrases shorter than MIN_PHRASE_LEN swars, and phrases whose full
n-gram occurs in more than MAX_PHRASE_DF DB raags. Rank the rest by idiosyncrasy = mean IDF
of their 2-/3-grams over the whole DB, so `G D P` outranks `G m P` at equal length.

    poetry run python phrases.py            # writes results/phrases.csv
"""

import csv
from dataclasses import dataclass, asdict

import numpy as np

import _bootstrap  # noqa: F401
import config as C
from utils import raagdb

SWAR = raagdb.SWAR_NAMES


@dataclass
class Phrase:
    raag: str            # dataset folder name
    idx: int             # position in the DB mukhyanga list
    swars: tuple         # collapsed swar indices 0..11
    octaves: tuple
    df: int              # DB raags containing the full phrase
    idf: float           # mean IDF of sub-n-grams
    turns: int           # melodic direction changes (a crude complexity count)
    kept: bool

    @property
    def id(self):
        return f"{self.raag}#{self.idx}"

    @property
    def text(self):
        mark = {-1: ",", 0: "", 1: "`"}
        return " ".join(mark.get(o, "") + SWAR[s] for s, o in zip(self.swars, self.octaves))


def _collapse_with_oct(s, o):
    out = []
    for pair in zip(s, o):
        if not out or out[-1][0] != pair[0]:
            out.append(pair)
    return tuple(p[0] for p in out), tuple(p[1] for p in out)


def _turns(swars, octaves):
    pitch = np.array(swars) + 12 * np.array(octaves)
    d = np.sign(np.diff(pitch))
    return int(np.sum(d[1:] != d[:-1]))


def catalogue(raag_names):
    df, n_db = raagdb.ngram_document_frequency(2, 12)
    lo, hi = C.NGRAM_RANGE
    out = []
    for name, r in sorted(raagdb.dataset_raags(raag_names).items()):
        for i, (s, o) in enumerate(zip(r.phrases, r.phrase_octaves)):
            s, o = _collapse_with_oct(s, o)
            grams = [s[j:j + n] for n in range(lo, hi + 1) for j in range(len(s) - n + 1)]
            idf = float(np.mean([np.log(n_db / max(df.get(g, 1), 1)) for g in grams])) if grams else 0.0
            dfull = df.get(s, 1)
            kept = len(s) >= C.MIN_PHRASE_LEN and dfull <= C.MAX_PHRASE_DF
            out.append(Phrase(name, i, s, o, dfull, round(idf, 3), _turns(s, o), kept))
    return out


def kept_phrases(raags=None):
    import contour
    names = sorted({c.raag for c in contour.clips().values()})
    return [p for p in catalogue(names) if p.kept and (raags is None or p.raag in raags)]


if __name__ == "__main__":
    import contour
    names = sorted({c.raag for c in contour.clips().values()})
    cat = catalogue(names)
    C.RESULTS_DIR.mkdir(exist_ok=True)
    with open(C.PHRASES_CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "raag", "phrase", "len", "df", "idf", "turns", "kept"])
        for p in sorted(cat, key=lambda p: (p.raag, -p.kept, -p.idf)):
            w.writerow([p.id, p.raag, p.text, len(p.swars), p.df, p.idf, p.turns, int(p.kept)])
    kept = [p for p in cat if p.kept]
    print(f"{len(cat)} phrases, {len(kept)} kept -> {C.PHRASES_CSV}")
    for p in sorted(kept, key=lambda p: -p.idf):
        if p.raag in C.FOCUS_RAAGS:
            print(f"  {p.id:20s} {p.text:26s} df={p.df:2d} idf={p.idf:.2f} turns={p.turns}")
