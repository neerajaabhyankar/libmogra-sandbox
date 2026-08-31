"""libmogra's raag database, reshaped into what a phrase matcher needs.

Everything downstream works in one 12-symbol alphabet: the swar indices 0..11 of
`libmogra.datatypes.Swar` (S r R g G m M P d D n N). Saptak marks in the DB (`,N`, `` `S ``)
are stripped here — an octave error in a pitch tracker is common enough that matching on
octave would cost more hits than it buys — but kept alongside as `octave` arrays in case a
later method wants them.
"""

import re
from functools import lru_cache

from libmogra.datatypes import Swar, SAPTAK_MARKS
from libmogra.raagfinder.parse import RAAG_DB

# The dataset's CamelCase folder names -> RAAG_DB keys. Fuzzy matching gets 48/50 right;
# these two it gets wrong or ambiguous, so they are pinned.
FOLDER_OVERRIDES = {
    "Sarang": "sarang (brindavani sarang)",
    "KaushikDhwani": "kaushik dhwani (bhinn shadj)",
}

SWAR_NAMES = [s.name for s in sorted(Swar, key=lambda s: s.value)]


def folder_to_db_key(folder):
    if folder in FOLDER_OVERRIDES:
        return FOLDER_OVERRIDES[folder]
    spaced = re.sub(r"(?<!^)(?=[A-Z])", " ", folder).lower()
    if spaced in RAAG_DB:
        return spaced
    import rapidfuzz

    return rapidfuzz.process.extractOne(spaced, [k for k in RAAG_DB if k not in ("all", "none")])[0]


def parse_swar(token):
    """'`S' -> (0, +1); ',n' -> (10, -1); 'M' -> (6, 0). Returns (swar_index, octave)."""
    token = token.strip()
    if not token:
        return None
    mark, name = "", token
    while name and name[0] in ",`":
        mark += name[0]
        name = name[1:]
    if not name:
        return None
    # SAPTAK_MARKS is keyed by the mark string itself (",,", ",", "", "`", "``")
    octave = SAPTAK_MARKS.get(mark, 0)
    if name not in Swar.__members__:
        upper = name.upper()
        if upper in Swar.__members__:  # DB has stray 'p'/'s'
            name = upper
        else:
            return None
    return Swar[name].value, octave


def parse_phrase(tokens):
    """['`S','n','d'] -> ([0,10,9], [1,0,0]) as (swars, octaves)."""
    parsed = [parse_swar(t) for t in tokens]
    parsed = [p for p in parsed if p is not None]
    return [p[0] for p in parsed], [p[1] for p in parsed]


def collapse(seq):
    """Drop consecutive duplicates: [4,4,5,5,4] -> [4,5,4]."""
    out = []
    for x in seq:
        if not out or out[-1] != x:
            out.append(x)
    return out


class Raag:
    """One DB entry, pre-folded into swar indices."""

    def __init__(self, key):
        self.key = key
        e = RAAG_DB[key]
        self.name = e.get("name", key)
        self.thaat = e.get("thaat")
        self.aaroha, self.aaroha_oct = parse_phrase(e.get("aaroha", []))
        self.avaroha, self.avaroha_oct = parse_phrase(e.get("avaroha", []))
        self.nyas = set()
        for f in ("aarohi_nyas", "avarohi_nyas"):
            s, _ = parse_phrase(e.get(f, []))
            self.nyas |= set(s)
        self.vaadi = parse_swar(e.get("vaadi") or "")
        self.samvaadi = parse_swar(e.get("samvaadi") or "")

        # mukhyanga: list of phrases, blank entries dropped, consecutive repeats collapsed
        self.phrases, self.phrase_octaves = [], []
        for ph in e.get("mukhyanga", []):
            s, o = parse_phrase(ph)
            if len(s) >= 2:
                self.phrases.append(s)
                self.phrase_octaves.append(o)

        # the raag's swar set (what may legitimately be sung)
        self.scale = set(self.aaroha) | set(self.avaroha)
        for p in self.phrases:
            self.scale |= set(p)

    @property
    def phrase_strings(self):
        return ["".join(SWAR_NAMES[s] for s in p) for p in self.phrases]

    def __repr__(self):
        return f"Raag({self.name}, scale={sorted(self.scale)}, {len(self.phrases)} phrases)"


@lru_cache(maxsize=1)
def all_raags():
    """Every DB entry, keyed by DB key. Used for IDF statistics over the whole database."""
    return {k: Raag(k) for k in RAAG_DB if k not in ("all", "none")}


@lru_cache(maxsize=1)
def _default_names():
    """The class list of the configured dataset, from its `tonics.csv`."""
    from . import config, dataset

    return tuple(dataset.raag_names(tonics_csv=config.tonics_csv()))


def dataset_raags(names=None):
    """`{class_name: Raag}` for the raags in play.

    `names` is the class list. Pass it explicitly wherever you have it -- a caller that
    already knows its labels should not make this function go and find them.

    Left None, it reads them from the configured dataset's `tonics.csv`
    (`utils.config.dataset_dir`, override with `RAAG_DATASET_DIR`). It used to instead
    list the *directories* of the v0 dataset tree, which meant 240 MB of audio had to be
    on disk for a metrics function to learn 50 strings, and meant the class list came
    from a different revision than the one under evaluation.
    """
    return _raags(tuple(names) if names is not None else _default_names())


@lru_cache(maxsize=4)
def _raags(names):
    return {n: Raag(folder_to_db_key(n)) for n in names}


@lru_cache(maxsize=8)
def ngram_document_frequency(n_min=2, n_max=4, corpus="all"):
    """How many DB raags contain each swar n-gram, over mukhyanga + aaroha + avaroha.

    This is the IDF backbone: `M P` sits in half the database and should count for almost
    nothing, while `,d ,n S g m` is Malkauns and nothing else.
    """
    raags = all_raags() if corpus == "all" else dataset_raags()
    df = {}
    for r in raags.values():
        seen = set()
        for seq in list(r.phrases) + [r.aaroha, r.avaroha]:
            seq = collapse(seq)
            for n in range(n_min, n_max + 1):
                for i in range(len(seq) - n + 1):
                    seen.add(tuple(seq[i : i + n]))
        for g in seen:
            df[g] = df.get(g, 0) + 1
    return df, len(raags)


if __name__ == "__main__":
    ds = dataset_raags()
    print(f"{len(ds)} dataset raags, {len(all_raags())} DB raags")
    for folder, r in list(ds.items())[:5]:
        print(f"  {folder:16s} {r.key:30s} {r.phrase_strings}")
    df, n = ngram_document_frequency()
    common = sorted(df.items(), key=lambda kv: -kv[1])[:5]
    print("most common n-grams:", [("".join(SWAR_NAMES[s] for s in g), c) for g, c in common])
