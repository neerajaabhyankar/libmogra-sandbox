"""The libmogra raag database as a (n_raags, 12) swar-occupancy table.

Used only when *training*: the head's templates start here instead of at random, so a raag
with 18 training clips inherits a usable reference profile on epoch zero. Inference reads
the trained templates out of the checkpoint, so a released model does not need libmogra.

The number for raag i, swar s is how often swar s appears across raag i's characteristic
phrases (mukhyanga) plus its aaroha and avaroha, normalised per raag. It is the field that
separates pairs which scale, vaadi and samvaadi cannot -- Bageshree and Bheempalasi share
all three and differ here by 0.43 in L1, in the musically correct direction (Bageshree
weakens Pa and leans on Dha).
"""

import re

import numpy as np

#: Dataset folder name -> libmogra key, for the two that fuzzy matching gets wrong.
OVERRIDES = {"Sarang": "sarang (brindavani sarang)",
             "KaushikDhwani": "kaushik dhwani (bhinn shadj)"}


def _db_key(folder, db):
    if folder in OVERRIDES:
        return OVERRIDES[folder]
    spaced = re.sub(r"(?<!^)(?=[A-Z])", " ", folder).lower()
    if spaced in db:
        return spaced
    import rapidfuzz

    return rapidfuzz.process.extractOne(
        spaced, [k for k in db if k not in ("all", "none")])[0]


def _swar_index(token, swars):
    """'`S' -> 0, ',n' -> 10, 'M' -> 6. Saptak marks are stripped: an octave error is
    common enough in pitch tracking that matching on register costs more than it buys."""
    name = str(token).strip().lstrip(",`")
    return swars.get(name)


def occupancy(raag_folders):
    """(len(raag_folders), 12) row-stochastic swar occupancy, in the order given."""
    from libmogra.datatypes import Swar
    from libmogra.raagfinder.parse import RAAG_DB

    swars = {s.name: s.value for s in Swar}
    out = np.zeros((len(raag_folders), 12), dtype=np.float64)
    for i, folder in enumerate(raag_folders):
        entry = RAAG_DB[_db_key(folder, RAAG_DB)]
        parse = lambda ph: [s for s in (_swar_index(t, swars) for t in ph) if s is not None]
        # a "phrase" of one swar carries no melodic information; the aaroha and avaroha
        # are always counted, whatever their length
        phrases = [p for p in map(parse, entry.get("mukhyanga", [])) if len(p) >= 2]
        for phrase in phrases + [parse(entry.get("aaroha", [])),
                                 parse(entry.get("avaroha", []))]:
            for s in phrase:
                out[i, s] += 1.0
    return (out / np.maximum(out.sum(axis=1, keepdims=True), 1e-12)).astype(np.float32)
