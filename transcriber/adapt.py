"""A small head that turns per-frame feature blocks into a lead pitch track (plan.md § Adaptation).

    fit(items)            -> Head trained on items = [{"blocks": [arrays], "hz": target, "mask": bool}]
    predict(head, blocks) -> Track on the 10 ms grid

Targets per frame: `hz` > 0 = the lead's pitch, 0 = silent; `mask` False = no label (ignored).
Loss: cross-entropy against a soft label -- a Gaussian over the 20-cent pitch bins around the
target pitch, or the extra "silent" class. Nothing here knows about raags or notation.
"""

import numpy as np
import torch
from torch import nn

from . import config as C
from .contract import Track
from .features import bins_hz

A = C.ADAPT


class Head(nn.Module):
    """Project each block to `width`, sum, then dilated 1-D convolutions with residuals."""

    def __init__(self, dims):
        super().__init__()
        w = A["width"]
        self.proj = nn.ModuleList(nn.Linear(d, w) for d in dims)
        self.convs = nn.ModuleList(
            nn.Conv1d(w, w, A["kernel"], padding=2 ** i * (A["kernel"] // 2), dilation=2 ** i)
            for i in range(A["layers"]))
        self.out = nn.Linear(w, A["n_bins"] + 1)        # last class = silent

    def forward(self, blocks):                           # blocks: list of (B, T, d)
        h = sum(p(b) for p, b in zip(self.proj, blocks)).transpose(1, 2)
        for c in self.convs:
            h = h + torch.nn.functional.gelu(c(h))
        return self.out(h.transpose(1, 2))               # (B, T, n_bins + 1)


def soft_labels(hz):
    """(T, n_bins + 1) target distributions; zeros for 0 Hz are replaced by the silent class."""
    y = np.zeros((len(hz), A["n_bins"] + 1), np.float32)
    v = hz > 0
    b = 1200 * np.log2(hz[v] / A["fmin_hz"]) / A["cents_per_bin"]
    g = np.exp(-0.5 * ((np.arange(A["n_bins"])[None] - b[:, None]) / A["label_sigma_bins"]) ** 2)
    y[v, :-1] = g / g.sum(1, keepdims=True)
    y[~v, -1] = 1.0
    return y


def _crops(items, rng):
    """One batch of random `crop`-frame windows (items shorter than a crop are zero-padded)."""
    xs, ys, ms = [], [], []
    for it in rng.choice(items, A["batch"]):
        n, L = len(it["hz"]), A["crop"]
        a = int(rng.integers(0, max(1, n - L)))
        pad = lambda x: np.pad(x[a:a + L], [(0, L - len(x[a:a + L]))] + [(0, 0)] * (x.ndim - 1))
        xs.append([pad(b) for b in it["blocks"]]); ys.append(pad(it["y"])); ms.append(pad(it["mask"]))
    t = lambda x: torch.from_numpy(np.stack(x))
    return [t([x[i] for x in xs]) for i in range(len(xs[0]))], t(ys), t(ms)


def fit(items, log=None):
    torch.manual_seed(A["seed"])
    rng = np.random.default_rng(A["seed"])
    items = [dict(it, y=soft_labels(it["hz"])) for it in items]
    head = Head([b.shape[1] for b in items[0]["blocks"]])
    opt = torch.optim.Adam(head.parameters(), lr=A["lr"])
    for step in range(A["steps"]):
        x, y, m = _crops(items, rng)
        logp = torch.log_softmax(head(x), -1)
        loss = -((logp * y).sum(-1) * m).sum() / m.sum().clamp(min=1)
        opt.zero_grad(); loss.backward(); opt.step()
        if log and step % 500 == 0:
            log(f"    step {step}: loss {loss.item():.3f}")
    return head.eval()


def predict(head, blocks):
    with torch.no_grad():
        p = torch.softmax(head([torch.from_numpy(b)[None] for b in blocks]), -1)[0].numpy()
    voiced = p[:, -1] < 1 - A["voiced_threshold"]
    k = p[:, :-1].argmax(1)
    lo, hi = np.maximum(k - 2, 0), np.minimum(k + 3, A["n_bins"])
    hz = bins_hz()
    est = np.array([np.exp(np.average(np.log(hz[a:b]), weights=p[t, a:b] + 1e-9))
                    for t, (a, b) in enumerate(zip(lo, hi))])
    return Track(np.where(voiced, est, 0.0).astype(np.float32), A["hop_s"], 1 - p[:, -1])
