"""The naive melody-only feature: a tonic-referenced, octave-folded pitch histogram.

This is M11's fingerprint from ../motif-classifier -- a 120-bin histogram of CREPE's f0
track, in cents against the clip's **annotated** tonic, circularly smoothed and
dynamic-range compressed. It is deliberately the *naive* feature: no notes, no n-grams, no
tonic search, nothing the symbolic pipeline does after this point.

It is used twice, which is why it lives here rather than in a script:

    scripts/01_probe_representations.py   as a frozen representation (logreg CV 0.434)
    scripts/10_train.py --melody          concatenated onto a CQT model's pooled feature

Both read the same cache, so the vector a network trains on is the same vector the probe
scored -- if the hybrid fails to beat the probe, that is a fact about the combination and
not about two different features sharing a name.
"""

import numpy as np

from .paths import CACHE, MOTIF_DIR, add_sibling_paths

#: The parameters M11 settled on; also the cache key.
DEFAULTS = dict(tracker="crepe", n_bins=120, smooth=1.0, power=0.5)


def histogram(clips, tracker="crepe", n_bins=120, smooth=1.0, power=0.5):
    """(n_clips, n_bins) pitch histograms, in clip order.

    Reuses ../motif-classifier's extracted pitch tracks and its own `fold_histogram`, so
    the only things this project contributes are the tonic lookup and the clip list. A
    clip with no track gets a zero row rather than being dropped -- the caller's indexing
    stays aligned, and a zero row is a clip the melody branch can say nothing about.
    """
    add_sibling_paths()
    from methods.m11_histogram import fold_histogram

    npz = MOTIF_DIR / "cache" / f"notes_{tracker}_v1.1.npz"
    if not npz.exists():
        raise FileNotFoundError(f"{npz} not found -- this feature reuses motif-classifier's "
                                f"pitch tracks; extract them there first")
    out, missing = [], 0
    with np.load(npz, allow_pickle=True) as z:
        keys = set(z.files)
        for c in clips:
            if f"{c.clip_id}|f0" not in keys:
                missing += 1
                out.append(np.zeros(n_bins))
                continue
            f0 = np.asarray(z[f"{c.clip_id}|f0"], dtype=float)
            n = int(z[f"{c.clip_id}|meta"][1])
            voiced = np.unpackbits(z[f"{c.clip_id}|voiced"])[:n].astype(bool)
            f0 = f0[:n][voiced]
            cents = 1200.0 * np.log2(np.clip(f0, 1e-6, None) / c.tonic_hz)
            out.append(fold_histogram(cents, None, n_bins, smooth, power))
    if missing:
        print(f"    WARNING {missing}/{len(clips)} clips missing from {npz.name}")
    return np.stack(out).astype(np.float32)


def cached(clips, force=False, **kw):
    """`histogram(clips)`, memoised on disk by clip id.

    The store is keyed by clip id rather than by the clip *list*, so asking for the test
    clips after the train clips extends the file instead of overwriting it -- the
    alternative recomputes 1810 histograms every time the caller's split changes.
    """
    params = {**DEFAULTS, **kw}
    path = CACHE / "melody" / ("_".join(f"{k}-{v}" for k, v in sorted(params.items())) + ".npz")

    store = {}
    if path.exists() and not force:
        with np.load(path, allow_pickle=True) as z:
            store = {str(k): v for k, v in zip(z["clip_ids"], z["X"])}
    todo = [c for c in clips if c.clip_id not in store]
    if todo:
        store.update(zip([c.clip_id for c in todo], histogram(todo, **params)))
        path.parent.mkdir(parents=True, exist_ok=True)
        ids = sorted(store)
        np.savez(path, X=np.stack([store[i] for i in ids]), clip_ids=np.array(ids))
        print(f"    melody: built {len(todo)} histograms -> {path.name} ({len(ids)} cached)")
    return np.stack([store[c.clip_id] for c in clips]).astype(np.float32)


def by_clip_id(clips, **kw):
    """{clip_id: (n_bins,)} -- the form the datasets want, since a Dataset is indexed by
    position in *its* clip list and several datasets share one cache."""
    return dict(zip([c.clip_id for c in clips], cached(clips, **kw)))
