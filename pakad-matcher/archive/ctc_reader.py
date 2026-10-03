"""S12: a learned reader -- a small network trained on the notation corpus to transcribe swars.

    poetry run python ctc_reader.py            # cross-validated misread, same folds as fit_reader
    poetry run python ctc_reader.py --save     # also train on all notation -> results/ctc_reader/

Why: the heuristic reader's misread is stuck near 0.56 however its constants are set
(DATA.md, "reading over-segments"): where a note starts is the problem, and that is exactly what
a hand-written segmentation rule gets wrong. Here nothing about segmentation is written down.

  input     per frame: tonic-relative pitch class (soft bins over the octave), register,
            slope, voiced flag -- from the same Melodia contour as everything else
  network   2 conv layers -> 2-layer bidirectional GRU -> 12 swars (x 3 octaves if `octaves`) + blank
  loss      per frame: the notated swar the notation aligner places there ("no note" on transits);
            plus `ctc_weight` x CTC on the bare sequence (0: it stalled and read worse, S12)
  decode    greedy; optionally restricted to a scale (`allowed`)

Result (S12): held-out misread 0.618 vs the tuned heuristic's 0.559 -- not adopted.

Augmentation: time-stretch and a small global detune. Early stopping on recordings held out of
the training side, so the outer cross-validation never informs training. Seeds are ensembled.

Terms: [DATA.md § Glossary](DATA.md#glossary).
"""

import argparse

import numpy as np
import torch
from torch import nn

import _bootstrap  # noqa: F401
import config as C
import corpus
import decode
from metrics import edit_ops

P = C.CTC
N_CLS = 12 * P["octaves"]       # swar (+ 12 * (octave + 1) when octaves are classes)
BLANK = N_CLS


def features(cents):
    """(T, F) float32 from a tonic-relative cents contour (NaN = unvoiced)."""
    c = np.asarray(cents, float)
    v = ~np.isnan(c)
    cc = np.where(v, c, 0.0)
    centres = np.arange(P["pc_bins"]) * 1200.0 / P["pc_bins"]
    d = (cc[:, None] - centres[None, :] + 600.0) % 1200.0 - 600.0
    pc = np.exp(-0.5 * (d / P["pc_sigma"]) ** 2) * v[:, None]
    slope = np.diff(cc, prepend=cc[:1]) * (v & np.roll(v, 1))
    return np.column_stack([pc, np.clip(cc / 1200.0, -1.5, 2.0) * v,
                            np.clip(slope / 50.0, -3, 3), v]).astype(np.float32)


def labels(st):
    if P["octaves"] == 1:
        return [s % 12 for s in st["swars"]]
    return [s % 12 + 12 * (min(max(o, -1), 1) + 1) for s, o in zip(st["swars"], st["octaves"])]


class Net(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        h = P["hidden"]
        self.conv = nn.Sequential(nn.Conv1d(n_in, h, 5, padding=2), nn.ReLU(),
                                  nn.Conv1d(h, h, 5, stride=P["stride"], padding=2), nn.ReLU())
        self.rnn = nn.GRU(h, h, P["layers"], batch_first=True, bidirectional=True,
                          dropout=P["dropout"])
        self.out = nn.Linear(2 * h, N_CLS + 1)

    def forward(self, x):                                   # (B, T, F) -> (B, T, C) log-probs
        y = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return self.out(self.rnn(y)[0]).log_softmax(-1)


_FRAMES = {}


def frame_labels(st):
    """Per input frame: the class of the notated swar sounding there (from the notation aligner),
    BLANK on transits, -100 (ignored) where the notes were only evenly spaced; and which notated
    note each frame belongs to (-1 = none)."""
    key = id(st)
    if key not in _FRAMES:
        if st["method"] != "align":
            _FRAMES[key] = (np.full(len(st["cents"]), -100), np.full(len(st["cents"]), -1))
        else:
            kinds, _ = decode.align(st["cents"], st["swars"], st["hop"],
                                    params=dict(C.NOTATE_MATCH), free_edges=True)
            lab = np.array(labels(st) + [BLANK])            # kinds == -1 -> BLANK
            _FRAMES[key] = (lab[np.where(kinds >= 0, kinds, len(lab) - 1)], kinds)
    return _FRAMES[key]


def _augment(st, rng):
    """Time-stretched, detuned contour and its frame labels at the network's frame rate."""
    n = len(st["cents"])
    f = rng.uniform(*P["tempo"]) if rng else 1.0
    idx = np.clip(np.round(np.arange(0, n, f)).astype(int), 0, n - 1)
    cents = st["cents"][idx] + (rng.normal(0, P["tuning_sd"]) if rng else 0.0)
    lab, note = frame_labels(st)
    lab, nt = lab[idx][::P["stride"]].copy(), note[idx][::P["stride"]]
    lab[1:][(nt[1:] != nt[:-1]) & (nt[:-1] >= 0) & (lab[1:] == lab[:-1])] = BLANK   # `R R`
    return cents, lab


def _ctc(net, items, rng=None):
    """CTC on the notated sequence + `frame_weight` x per-frame cross-entropy on the alignment."""
    aug = [_augment(s, rng) for s in items]
    xs = [torch.from_numpy(features(c)) for c, _ in aug]
    x = nn.utils.rnn.pad_sequence(xs, batch_first=True)
    xl = torch.tensor([-(-len(a) // P["stride"]) for a in xs])  # lengths after the stride
    ys = [labels(s) for s in items]
    lp = net(x)
    ctc = nn.functional.ctc_loss(lp.transpose(0, 1), torch.tensor(sum(ys, [])), xl,
                                 torch.tensor([len(y) for y in ys]), blank=BLANK,
                                 zero_infinity=True)
    fl = nn.utils.rnn.pad_sequence([torch.from_numpy(l[:n]) for (_, l), n in zip(aug, xl)],
                                   batch_first=True, padding_value=-100)
    ce = nn.functional.nll_loss(lp[:, :fl.shape[1]].reshape(-1, N_CLS + 1), fl.reshape(-1).long(),
                                ignore_index=-100)
    return P["ctc_weight"] * ctc + P["frame_weight"] * ce


def train(stretches, seed):
    """One network; early-stopped on recordings held out of `stretches`."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    recs = sorted({s["recording"] for s in stretches})
    inner = set(rng.choice(recs, max(1, int(len(recs) * P["inner_frac"])), replace=False))
    fit = [s for s in stretches if s["recording"] not in inner]
    hold = [s for s in stretches if s["recording"] in inner]
    net = Net(features(np.zeros(2)).shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=P["lr"], weight_decay=P["weight_decay"])
    by_len = sorted(fit, key=lambda s: len(s["cents"]))       # similar lengths share a batch
    batches = [by_len[k:k + P["batch"]] for k in range(0, len(by_len), P["batch"])]
    best, best_state, wait = np.inf, None, 0
    for _ep in range(P["epochs"]):
        net.train()
        for b in rng.permutation(len(batches)):
            opt.zero_grad()
            loss = _ctc(net, batches[b], rng)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
        net.eval()
        with torch.no_grad():
            h = misread(hold, [net])
        if h < best - 1e-4:
            best, wait = h, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            wait += 1
            if wait >= P["patience"]:
                break
    net.load_state_dict(best_state)
    net.eval()
    return net


def read(cents, nets, allowed=None, frames=False):
    """Swar sequence (mod 12) -- or, with frames=True, [(swar, cents, f0, f1)] per note, where
    `cents` is the note's pitch on the contour (octave included) and f0:f1 its input frames."""
    with torch.no_grad():
        x = torch.from_numpy(features(cents))[None]
        lp = torch.stack([n(x)[0] for n in nets]).mean(0).numpy()
    if allowed is not None:
        keep = np.zeros(N_CLS + 1, bool)
        keep[[s + 12 * o for s in allowed for o in range(P["octaves"])]] = True
        keep[BLANK] = True
        lp = np.where(keep, lp, -np.inf)
    best, r = lp.argmax(1), P["stride"]
    notes = []
    for t, k in enumerate(best):
        if k == BLANK:
            continue
        if notes and notes[-1][3] == r * t and best[t - 1] == k:
            notes[-1][3] = r * (t + 1)
        else:
            notes.append([k % 12, None, r * t, r * (t + 1)])
    if not frames:
        return [n[0] for n in notes]
    out = []
    for s, _, f0, f1 in notes:
        v = np.asarray(cents[f0:f1], float)
        if np.isnan(v).all():
            continue
        c = float(np.nanmedian(v))
        out.append((s, 100 * s + 1200 * round((c - 100 * s) / 1200), f0, f1))
    return out


def misread(stretches, nets, totals=False):
    ops, n, n_read = np.zeros(3, int), 0, 0
    for st in stretches:
        human = [s % 12 for s in st["swars"]]
        seq = read(st["cents"], nets)
        ops += np.array(edit_ops(human, seq))
        n += len(human); n_read += len(seq)
    return (ops, n, n_read) if totals else ops.sum() / max(n, 1)


def cross_validate(k=4):
    st, fold = corpus.folds(k)
    fold = np.array(fold)
    tot, n, n_read = np.zeros(3, int), 0, 0
    by_kind = {}
    for f in range(k):
        train_ = [s for s, g in zip(st, fold) if g != f]
        held = [s for s, g in zip(st, fold) if g == f]
        nets = [train(train_, seed) for seed in P["seeds"]]
        ops, m, r = misread(held, nets, totals=True)
        tot += ops; n += m; n_read += r
        for kind in ("alap", "madhya", "taan"):
            o, m2, _ = misread([s for s in held if s["kind"] == kind], nets, totals=True)
            b = by_kind.setdefault(kind, [np.zeros(3, int), 0]); b[0] += o; b[1] += m2
        print(f"  fold {f + 1}/{k}: held-out misread {ops.sum() / m:.3f}", flush=True)
    print(f"\nlearned reader, held out: read {n_read}/{n} notated   sub {tot[0]}  del {tot[1]}  "
          f"ins {tot[2]}   misread {tot.sum() / n:.3f}")
    for kind, (o, m) in by_kind.items():
        print(f"  {kind:7s} misread {o.sum() / max(m, 1):.3f}  (sub {o[0]} del {o[1]} ins {o[2]})")
    return tot.sum() / n


def save():
    C.CTC_DIR.mkdir(parents=True, exist_ok=True)
    st = corpus.stretches()
    for seed in P["seeds"]:
        torch.save(train(st, seed).state_dict(), C.CTC_DIR / f"seed{seed}.pt")
    print(f"-> {C.CTC_DIR}")


def load():
    nets = []
    for seed in P["seeds"]:
        net = Net(features(np.zeros(2)).shape[1])
        net.load_state_dict(torch.load(C.CTC_DIR / f"seed{seed}.pt"))
        nets.append(net.eval())
    return nets


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(4)
    cross_validate()
    if a.save:
        save()
