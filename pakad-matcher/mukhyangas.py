"""`neeraja_mukhyangas.json` -- the hand-picked phrases, as `phrases.Phrase` objects.

This file, not libmogra's RaagDB, decides what gets annotated: entries may be shortened,
modified or new. Swars are validated against the raag's scale and a mismatch is loud.
"""

import json

import _bootstrap  # noqa: F401
import config as C
import phrases
from utils import raagdb


def load(only_annotate=True, raags=None):
    entries = json.loads(C.MUKHYANGAS_JSON.read_text())["phrases"]
    out = []
    for e in entries:
        if only_annotate and not e.get("annotate", True):
            continue
        if raags and e["raag"] not in raags:
            continue
        swars, octaves = raagdb.parse_phrase(e["phrase"].split())
        if len(swars) != len(e["phrase"].split()):
            raise ValueError(f"{e['id']}: could not parse {e['phrase']!r}")
        raag = raagdb.dataset_raags([e["raag"]])[e["raag"]]
        outside = {raagdb.SWAR_NAMES[s] for s in swars} - {raagdb.SWAR_NAMES[s] for s in raag.scale}
        if outside:
            print(f"WARNING {e['id']}: {sorted(outside)} not in {e['raag']}'s scale")
        _, ident = e["id"].split("#")
        p = phrases.build(e["raag"], ident, swars, octaves)
        p.source, p.note = e.get("source"), e.get("note")
        out.append(p)
    return out


if __name__ == "__main__":
    for p in load():
        print(f"{p.id:22s} {p.text:16s} len {len(p.swars)}  df {p.df:2d}  idf {p.idf:.2f}  {p.source}")
