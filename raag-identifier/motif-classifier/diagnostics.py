"""Is the transcription good enough to match phrases against at all?

Before tuning anything, check the two things every method depends on and neither can fix:

* **Tonic sanity.** With a correct tonic, most of a clip's sung duration should land on
  swars the true raag actually contains. If that number is near chance (a random 7-of-12
  scale catches ~58 % of a uniform distribution), the tonic is wrong and no amount of
  phrase cleverness will help. Reported as `in_scale`, alongside the best value reachable
  by rotating the clip (`in_scale_best`) and which rotation that was.
* **Phrase reachability.** How often does the true raag's mukhyanga appear at all — exactly,
  or as a skip-tolerant n-gram? This is the ceiling on M1.
"""

import argparse
from collections import Counter

import numpy as np

from features import build_features
import _bootstrap  # noqa: F401  (puts raag-identifier/ on sys.path)
from utils.raagdb import collapse, dataset_raags
from represent import Params, build_clips


def scale_masks(raags):
    m = {}
    for folder, r in raags.items():
        v = np.zeros(12)
        for s in r.scale:
            v[s] = 1.0
        m[folder] = v
    return m


def report(p: Params, split="train", max_skip=1):
    raags = dataset_raags()
    masks = scale_masks(raags)
    clips = [c for c in build_clips(p) if c.split == split]
    feats = build_features([c for c in clips], max_skip=max_skip)

    in_scale, in_scale_best, best_shifts = [], [], []
    exact_hit, ngram_hit, n_notes = [], [], []
    for f in feats:
        if f.n_notes < 2:
            continue
        n_notes.append(f.n_notes)
        mask = masks[f.clip.raag]
        vals = []
        for k in range(12):
            d = f.rot_unigram_dur(k)
            vals.append(float(mask @ d / max(d.sum(), 1e-9)))
        in_scale.append(vals[0])
        in_scale_best.append(max(vals))
        best_shifts.append(int(np.argmax(vals)))

        r = raags[f.clip.raag]
        seq = "".join(chr(65 + s) for s in f.clip.swars)
        phrases = [tuple(collapse(x)) for x in r.phrases]
        exact_hit.append(any("".join(chr(65 + s) for s in ph) in seq for ph in phrases))
        grams = set()
        for n in range(2, 5):
            grams |= set(f.ngrams.get(n, {}))
        ngram_hit.append(
            np.mean([sum(tuple(ph[i : i + 2]) in grams for i in range(len(ph) - 1)) / max(len(ph) - 1, 1)
                     for ph in phrases]) if phrases else 0.0
        )

    print(f"--- {p}  split={split}  ({len(n_notes)} usable clips of {len(feats)})")
    print(f"  notes/clip           median {np.median(n_notes):.0f}   mean {np.mean(n_notes):.1f}")
    print(f"  in-scale duration    {np.mean(in_scale):.3f}  (best rotation: {np.mean(in_scale_best):.3f})")
    print(f"  best rotation == 0   {np.mean(np.array(best_shifts) == 0):.3f}")
    print(f"  rotation histogram   {dict(sorted(Counter(best_shifts).items()))}")
    print(f"  >=1 exact mukhyanga  {np.mean(exact_hit):.3f}")
    print(f"  mukhyanga bigram cov {np.mean(ngram_hit):.3f}")
    return {
        "in_scale": float(np.mean(in_scale)),
        "in_scale_best": float(np.mean(in_scale_best)),
        "rot0_is_best": float(np.mean(np.array(best_shifts) == 0)),
        "exact_hit": float(np.mean(exact_hit)),
        "bigram_cov": float(np.mean(ngram_hit)),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    args = ap.parse_args()
    for src in ("hmm", "segment"):
        for tonic_mode in ("clip", "video"):
            for refine in (False, True):
                report(Params(note_source=src, tonic_mode=tonic_mode, tonic_refine=refine), split=args.split)
