"""Cheap probes: is there raag signal in each representation, before any fine-tuning?

Four hours of distilHuBERT is a bad way to discover that a representation is at chance. Every
probe here freezes the representation, fits only a linear model on top, and runs
video-grouped 5-fold CV over the 1810 training clips. The whole script is minutes, not hours.

It also serves as the **harness self-test**. `chroma_anchor` + `chi2` is this project's
reimplementation of M11 from ../motif-classifier -- an octave-folded pitch profile scored
against per-raag mean templates by chi-square, no musical knowledge at all -- which scored
**0.395 train CV** there. If the number here comes out near that, the splits, the labels,
the tonic and the metrics are all wired correctly. If it comes out at 0.02, the bug is in
this folder and not in any model.

Representations
    chroma_anchor    Sa-anchored CQT, octave-folded to 36 bins. Bin 0 is Sa exactly.
    chroma_fixed     the same CQT with a fixed fmin: absolute pitch. The tonic control --
                     the gap between these two *is* what the annotation buys, measured
                     without training anything.
    chroma_argmax    anchored, but only the loudest bin per frame votes -- closer to a
                     melody histogram, and much less contaminated by harmonics (a note's
                     3rd harmonic lands on its fifth, its 5th on its major third).
    hubert_frozen    mean-pooled last hidden state of ntu-spml/distilhubert, off the shelf.
    jeevster_frozen  the 300-d global-average-pooled feature of the Carnatic ResNet.

Classifiers
    chi2     nearest per-raag mean profile under chi-square distance. Not fitted beyond the
             class means. ../motif-classifier found chi2 beat cosine decisively, so it is
             the default for anything non-negative.
    cosine   the same with cosine distance -- kept only to reproduce that finding here.
    logreg   multinomial logistic regression on standardised features. The honest linear
             probe, and the only option for the embedding representations.

    poetry run python scripts/01_probe_representations.py
    poetry run python scripts/01_probe_representations.py --reps chroma_anchor --clfs chi2
"""

import argparse
import json
import time

import numpy as np

import _bootstrap  # noqa: F401
from common import audio, melody, metrics
from common.data import fold_indices, labels, load_clips, summarise
from common.paths import CACHE, RESULTS

EPS = 1e-9


# ----------------------------------------------------------------- representations


def _chroma(clip, tonic, mode):
    C = audio.db_to_amplitude(audio.cached_cqt(clip, tonic=tonic))
    if mode == "argmax":
        # one vote per frame for its loudest bin, weighted by how loud the frame is
        peak = C.argmax(axis=0)
        weight = C.max(axis=0)
        H = np.zeros(C.shape[0], dtype=np.float32)
        np.add.at(H, peak, weight)
        H = H.reshape(-1, audio.CQT_BINS_PER_OCTAVE).sum(axis=0)
    else:
        H = audio.fold_octaves(C).sum(axis=1)
    return (H / max(H.sum(), EPS)).astype(np.float32)


def _hubert_frozen(clips):
    import torch
    from transformers import AutoModel

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModel.from_pretrained("ntu-spml/distilhubert").to(device).eval()
    out = []
    with torch.no_grad():
        for i, c in enumerate(clips):
            y = audio.clip_tensor(c, audio.SR_HUBERT)
            y = (y - y.mean()) / (y.std() + 1e-7)
            h = model(torch.from_numpy(y)[None].to(device)).last_hidden_state
            out.append(h.mean(dim=1).squeeze(0).cpu().numpy())
            if (i + 1) % 200 == 0:
                print(f"    hubert {i + 1}/{len(clips)}", flush=True)
    return np.stack(out)


def _jeevster_frozen(clips):
    import torch

    from common.paths import JEEVSTER_DIR, RESNET_DIR, add_sibling_paths
    add_sibling_paths()
    from raag_resnet.configuration_raag_resnet import RaagResNetConfig
    from raag_resnet.modeling_raag_resnet import RaagResNetForAudioClassification

    model = RaagResNetForAudioClassification(RaagResNetConfig(num_labels=50))
    model.load_backbone_weights(JEEVSTER_DIR / "ckpts" / "best_ckpt.tar")
    model.eval()
    out = []
    with torch.no_grad():
        for i, c in enumerate(clips):
            y = audio.clip_tensor(c, audio.SR_JEEVSTER)
            x = torch.from_numpy(y)[None].repeat(2, 1)            # mono -> the 2ch it expects
            x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-5)
            out.append(model.backbone_forward(x[None]).squeeze(0).numpy())
            if (i + 1) % 200 == 0:
                print(f"    jeevster {i + 1}/{len(clips)}", flush=True)
    return np.stack(out)


REPRESENTATIONS = {
    "melody_hist": melody.cached,   # common/melody.py -- also the Stage 5 hybrid feature
    "chroma_anchor": lambda cs: np.stack([_chroma(c, "anchor", "energy") for c in cs]),
    "chroma_fixed": lambda cs: np.stack([_chroma(c, "none", "energy") for c in cs]),
    "chroma_argmax": lambda cs: np.stack([_chroma(c, "anchor", "argmax") for c in cs]),
    "hubert_frozen": _hubert_frozen,
    "jeevster_frozen": _jeevster_frozen,
}

#: which classifiers make sense for which representation (chi2 needs non-negative inputs)
NONNEGATIVE = {"melody_hist", "chroma_anchor", "chroma_fixed", "chroma_argmax"}


def features(name, clips, force=False):
    """Representation matrix (n_clips, d), cached to disk -- these are reused across the
    classifier sweep and across reruns."""
    p = CACHE / "probe" / f"{name}.npz"
    key = np.array([c.clip_id for c in clips])
    if p.exists() and not force:
        z = np.load(p, allow_pickle=True)
        if np.array_equal(z["clip_ids"], key):
            return z["X"]
    t0 = time.time()
    X = REPRESENTATIONS[name](clips).astype(np.float32)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, X=X, clip_ids=key)
    print(f"    built {name} {X.shape} in {time.time() - t0:.0f}s")
    return X


# ----------------------------------------------------------------- classifiers


def _template_scores(Xtr, ytr, Xva, n_classes, metric):
    """Nearest per-class mean profile. Returns (n_va, n_classes) scores, higher = better."""
    refs = np.stack([
        Xtr[ytr == c].mean(axis=0) if np.any(ytr == c) else np.full(Xtr.shape[1], EPS)
        for c in range(n_classes)
    ])
    refs = refs / np.maximum(refs.sum(axis=1, keepdims=True), EPS)
    Q = Xva / np.maximum(Xva.sum(axis=1, keepdims=True), EPS)
    if metric == "chi2":
        d = 0.5 * (((Q[:, None, :] - refs[None]) ** 2) / (Q[:, None, :] + refs[None] + EPS)).sum(-1)
        return -d
    if metric == "cosine":
        A = Q / np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), EPS)
        B = refs / np.maximum(np.linalg.norm(refs, axis=1, keepdims=True), EPS)
        return A @ B.T
    raise ValueError(metric)


def _logreg_scores(Xtr, ytr, Xva, n_classes):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(sc.transform(Xtr), ytr)
    S = np.full((len(Xva), n_classes), -1e3)
    S[:, clf.classes_] = clf.decision_function(sc.transform(Xva))
    return S


def cross_validate(X, clips, clf, n_folds=5, seed=0):
    """Video-grouped CV. Returns pooled metrics over the out-of-fold predictions."""
    n_classes = len(labels())
    y = np.array([c.label for c in clips])
    index = {c.clip_id: i for i, c in enumerate(clips)}
    oof = np.zeros((len(clips), n_classes))
    for _k, tr, va in fold_indices(clips, n_folds=n_folds, seed=seed):
        itr = np.array([index[c.clip_id] for c in tr])
        iva = np.array([index[c.clip_id] for c in va])
        if clf == "logreg":
            S = _logreg_scores(X[itr], y[itr], X[iva], n_classes)
        else:
            S = _template_scores(X[itr], y[itr], X[iva], n_classes, clf)
        oof[iva] = S
    return metrics.score(clips, oof)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", nargs="*", default=list(REPRESENTATIONS))
    ap.add_argument("--clfs", nargs="*", default=["chi2", "cosine", "logreg"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--force", action="store_true", help="rebuild cached representations")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    clips = load_clips("train")
    print(f"train: {summarise(clips)}; chance = {1 / len(labels()):.3f}")
    if args.dry_run:
        print(f"would probe {args.reps} x {args.clfs} with {args.folds}-fold grouped CV")
        return

    out_dir = RESULTS / "p0_probes"
    out_dir.mkdir(parents=True, exist_ok=True)
    table, rows_for_confusion = [], {}
    for rep in args.reps:
        X = features(rep, clips, force=args.force)
        for clf in args.clfs:
            if clf in ("chi2", "cosine") and rep not in NONNEGATIVE:
                continue
            t0 = time.time()
            m, rows = cross_validate(X, clips, clf, n_folds=args.folds)
            mus = metrics.musical(rows, temperature=metrics.calibrate_temperature(rows))
            table.append({"rep": rep, "clf": clf, "dim": int(X.shape[1]),
                          "seconds": round(time.time() - t0, 1), **m,
                          "mistake_affinity": mus.get("mistake_affinity"),
                          "mistake_affinity_chance": mus.get("mistake_affinity_chance"),
                          "tonic_explained": mus.get("tonic_explained")})
            rows_for_confusion[f"{rep}__{clf}"] = rows
            print(f"  {rep:16s} {clf:7s} {metrics.summary_line(m)}", flush=True)

    (out_dir / "probes.json").write_text(json.dumps(table, indent=2))
    print(f"\nwrote {out_dir / 'probes.json'}")

    best = max(table, key=lambda r: r["top1"])
    metrics.confusion(rows_for_confusion[f"{best['rep']}__{best['clf']}"],
                      out_dir / "confusion_best.png",
                      title=f"probe {best['rep']} / {best['clf']} — CV top-1 {best['top1']:.3f}")
    print(f"best probe: {best['rep']} / {best['clf']} at top-1 {best['top1']:.3f}")


if __name__ == "__main__":
    main()
