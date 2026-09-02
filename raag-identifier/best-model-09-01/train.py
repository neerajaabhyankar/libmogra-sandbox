"""Reproduce this model from scratch: pinned dataset in, `weights/` out.

    python train.py                       # the released recipe: every training clip, 34 epochs
    python train.py --val-fraction 0.2    # hold a fifth out, and report on it
    python train.py --test                # also score the held-out test split at the end
    python train.py --limit 60 --epochs 2 # smoke test in a few minutes

Seeded end to end (torch, numpy and the split), so two runs of the same command produce the
same weights up to non-determinism in the CQT/CREPE kernels.

**What the recipe is, and why each part of it.**

* The CQT network trains for a *fixed* number of epochs on every training clip. 34 is not
  arbitrary: a companion run that held out a fifth of the videos selected epoch 34 of 40 on
  validation top-1, and the released model reuses that number rather than early-stopping on
  data it is also fitting. Run with `--val-fraction 0.2` to watch that happen.
* Augmentation is a random roll of up to 2 CQT bins -- 22 cents at 36 bins per octave. It
  must stay *below* a semitone: the point is tuning drift between performances, and a roll
  of three bins would move a swar onto its neighbour and relabel the raag.
* Checkpoint selection, when there is a validation split, is on top-1 and not on validation
  loss. On this corpus a 50-class model's validation loss bottoms out within two epochs and
  then climbs while top-1 keeps improving for another twenty; selecting on loss picks a
  model barely above chance.
* The melody branch is a plain multinomial logistic regression. It is fitted on the same
  clips the network saw, so the fusion weight is not chosen with any help from data the
  network was validated on.

Cost: about 25 minutes on an M1 (10 for the feature cache, 15 for 34 epochs), plus the
dataset download. CREPE dominates the cache build; pass `--device mps` or `--device cuda`.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from raag_fusion import cqt_branch, data, db_templates, melody_branch

HERE = Path(__file__).resolve().parent

#: Chosen on the validation split of the companion run; see the module docstring.
DEFAULTS = dict(epochs=34, batch_size=16, lr=1e-3, weight_decay=1e-4, grad_clip=5.0,
                freq_jitter_bins=2, db_lam=0.3, seed=0,
                melody_weight=0.40, temperature_cqt=0.925, temperature_melody=2.360)


def pick_device(name="auto"):
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


# ------------------------------------------------------------------ the CQT branch


def train_cqt(X, y, raags, cfg, device, val=None, log=print):
    """X (n, 1, 144, 431) float32, y (n,) int -> a trained `cqt_branch.Net`."""
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    net = cqt_branch.Net(n_raags=len(raags), lam=cfg["db_lam"]).to(device)
    net.head.db_templates.copy_(
        torch.from_numpy(db_templates.occupancy(raags)).to(device))
    # the learned half of each template starts *at* the database, so epoch 0 is the prior
    with torch.no_grad():
        net.head.learned_logits.copy_(torch.log(net.head.db_templates + cqt_branch.EPS))

    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    rng = np.random.default_rng(cfg["seed"])
    best = {"top1": -1.0, "epoch": -1, "state": None}

    for epoch in range(cfg["epochs"]):
        net.train()
        t0, total = time.time(), 0.0
        order = rng.permutation(len(Xt))
        for i in range(0, len(order), cfg["batch_size"]):
            idx = order[i:i + cfg["batch_size"]]
            xb = Xt[idx].clone()
            if cfg["freq_jitter_bins"]:                 # tuning drift, under a semitone
                j = cfg["freq_jitter_bins"]
                for k in range(len(xb)):                # a separate shift per clip
                    xb[k] = torch.roll(xb[k], int(rng.integers(-j, j + 1)), dims=1)
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(net(xb.to(device)), yt[idx].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg["grad_clip"])
            opt.step()
            total += float(loss.detach()) * len(idx)

        line = f"  epoch {epoch:2d}  train_loss {total / len(order):.3f}"
        if val is not None:
            top1 = float((predict_logits(net, val[0], device).argmax(1) == val[1]).mean())
            line += f"  val_top1 {top1:.3f}"
            if top1 > best["top1"]:
                best = {"top1": top1, "epoch": epoch,
                        "state": {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}}
                line += "  <-"
        log(line + f"  [{time.time() - t0:.0f}s]")

    if best["state"] is not None:
        log(f"  best epoch {best['epoch']} at val top-1 {best['top1']:.3f}")
        net.load_state_dict(best["state"])
    return net


@torch.no_grad()
def predict_logits(net, X, device, batch_size=32):
    """Logits for a stack of windows, leaving the net in whatever mode it was in.

    Restoring the mode rather than assuming `train()` is not tidiness: this is called
    mid-training *and* from evaluation, and a net accidentally left in training mode fails
    on the next single-window forward pass, because batch norm cannot compute a batch
    statistic from one item.
    """
    was_training = net.training
    net.eval()
    out = [net(torch.from_numpy(X[i:i + batch_size]).to(device)).cpu().numpy()
           for i in range(0, len(X), batch_size)]
    net.train(was_training)
    return np.concatenate(out)


# ------------------------------------------------------------------ the melody branch


def fit_melody(H, y, n_raags):
    """Standardise + multinomial logistic regression -> a `melody_branch.LinearModel`."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(H)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(scaler.transform(H), y)
    coef = np.full((n_raags, H.shape[1]), 0.0)
    intercept = np.full(n_raags, -1e3)          # a raag absent from the fit can never win
    coef[clf.classes_] = clf.coef_
    intercept[clf.classes_] = clf.intercept_
    return melody_branch.LinearModel(scaler.mean_, scaler.scale_, coef, intercept)


# ------------------------------------------------------------------ putting it together


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=HERE / "weights", type=Path)
    ap.add_argument("--cache", default=HERE / "cache", type=Path)
    ap.add_argument("--val-fraction", type=float, default=0.0,
                    help="hold out this fraction of *videos* to select and calibrate on. "
                         "0 (the default) trains the released model on every clip")
    ap.add_argument("--test", action="store_true",
                    help="score the held-out test split once, at the end")
    ap.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    ap.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    ap.add_argument("--limit", type=int, default=None, help="at most N clips per raag per split, for a smoke test")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    device = pick_device(a.device)
    cfg = {**DEFAULTS, "epochs": a.epochs, "seed": a.seed}
    print(f"device={device}  seed={cfg['seed']}  epochs={cfg['epochs']}")

    clips = data.build_cache(a.cache, limit=a.limit, device=device)
    raags = sorted({c.raag for c in clips})
    label = {r: i for i, r in enumerate(raags)}
    train_clips = [c for c in clips if c.split == "train"]

    val_clips = []
    if a.val_fraction:
        train_clips, val_clips = data.grouped_split(train_clips, a.val_fraction, cfg["seed"])
    print(f"{len(raags)} raags | fit on {len(train_clips)} clips "
          f"({len({c.video for c in train_clips})} videos)"
          + (f" | validate on {len(val_clips)}" if val_clips else " | no held-out split"))

    X, H = data.load_features(a.cache, train_clips)
    y = np.array([label[c.raag] for c in train_clips])
    val = None
    if val_clips:
        Xv, Hv = data.load_features(a.cache, val_clips)
        val = (Xv, np.array([label[c.raag] for c in val_clips]))

    print("training the CQT branch")
    net = train_cqt(X, y, raags, cfg, device, val=val)
    print("fitting the melody branch")
    linear = fit_melody(H, y, len(raags))

    if val_clips:                       # calibrate and mix on data the model did not fit
        cfg.update(calibrate(net, linear, Xv, Hv, val[1], device))

    a.out.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in net.state_dict().items()}, a.out / "cqt_net.pt")
    linear.save(a.out / "melody_linear.npz")
    (a.out / "raags.json").write_text(json.dumps(raags, indent=1))
    (a.out / "config.json").write_text(json.dumps(
        {**{k: cfg[k] for k in ("melody_weight", "temperature_cqt", "temperature_melody",
                                "db_lam", "epochs", "seed")},
         "dataset": f"{data.DATASET_ID}@{data.REVISION}",
         "n_fit_clips": len(train_clips),
         "calibrated_on": f"{len(val_clips)} held-out clips" if val_clips
                          else "reused from the companion validation run"}, indent=1))
    print(f"wrote {a.out}")

    if a.test:
        evaluate_test(a, raags, label, device)


def calibrate(net, linear, Xv, Hv, yv, device):
    """Fit each branch's softmax temperature, then sweep the mixing weight. All on val."""
    from raag_fusion.identifier import _softmax

    dl = predict_logits(net, Xv, device)
    ml = np.stack([linear.scores(h) for h in Hv])

    def best_temperature(scores):
        best, best_nll = 1.0, np.inf
        for T in np.geomspace(0.01, 100.0, 60):
            p = np.stack([_softmax(s, T) for s in scores])
            nll = -np.log(np.clip(p[np.arange(len(yv)), yv], 1e-12, None)).mean()
            if nll < best_nll:
                best, best_nll = float(T), float(nll)
        return best

    t_cqt, t_mel = best_temperature(dl), best_temperature(ml)
    P = (np.stack([_softmax(s, t_cqt) for s in dl]),
         np.stack([_softmax(s, t_mel) for s in ml]))
    scores = {w: float((((1 - w) * P[0] + w * P[1]).argmax(1) == yv).mean())
              for w in np.round(np.arange(0.0, 1.01, 0.05), 2)}
    w = max(scores, key=scores.get)
    print(f"  calibration: T_cqt {t_cqt:.3f}  T_melody {t_mel:.3f}  weight {w:.2f} "
          f"(val top-1 {scores[w]:.3f}; branches alone {scores[0.0]:.3f} / {scores[1.0]:.3f})")
    return {"temperature_cqt": t_cqt, "temperature_melody": t_mel, "melody_weight": float(w)}


def evaluate_test(a, raags, label, device):
    """Score the held-out test split, once, through the *public* inference path.

    Deliberately not through the cached features the network trained on. Training sees the
    middle 20 s of each clip, because that is what a training example is; a user calling
    `predict` gets every 20 s window of their recording averaged. Those are different
    computations, and the number that belongs on the model card is the one describing what
    the user actually runs. This re-reads the audio to get it.
    """
    import io

    import librosa

    from raag_fusion import RaagIdentifier

    model = RaagIdentifier.load(a.out, device=device)
    y_true, y_pred, hits5, seen = [], [], 0, {}
    for raag, filename, blob, tonic_hz in data.stream():
        if not filename.startswith("test_"):
            continue
        seen[raag] = seen.get(raag, 0) + 1
        if a.limit and seen[raag] > a.limit:
            continue
        y, sr = librosa.load(io.BytesIO(blob), sr=None, mono=True)
        p = model.probabilities(y, sr, tonic_hz)
        rank = np.argsort(-p)
        y_true.append(label[raag])
        y_pred.append(int(rank[0]))
        hits5 += label[raag] in rank[:5]
        if len(y_true) % 25 == 0:
            print(f"  scored {len(y_true)}...", flush=True)

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    top1, top5 = float((y_pred == y_true).mean()), hits5 / len(y_true)
    print(f"\nTEST ({len(y_true)} clips, video-disjoint, full inference path): "
          f"top-1 {top1:.3f}  top-5 {top5:.3f}")
    (a.out / "test_metrics.json").write_text(json.dumps(
        {"n_clips": len(y_true), "top1": top1, "top5": top5,
         "path": "RaagIdentifier.probabilities on the raw audio"}, indent=1))
    confusion(y_true, y_pred, raags, HERE / "assets" / "confusion_test.png",
              f"held-out test, {len(y_true)} clips - top-1 {top1:.3f}")


def confusion(y_true, y_pred, raags, out_path, title=""):
    """The 50x50 matrix. It is the artifact that says *what* the model does -- which raags
    are solved, and which ones act as hubs that absorb everything near them."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.zeros((len(raags), len(raags)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    fig, ax = plt.subplots(figsize=(18, 16))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(len(raags)):
        for j in range(len(raags)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6,
                        color="white" if cm[i, j] > cm.max() * 0.6 else "black")
    ax.set_xticks(range(len(raags)), raags, rotation=90, fontsize=7)
    ax.set_yticks(range(len(raags)), raags, fontsize=7)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.6)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
