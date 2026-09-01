"""Objectives, including the ones that carry the libmogra database into training.

Plain cross-entropy says every wrong answer is equally wrong. That is false here and it is
also a waste: the database knows that Bageshree and Bheempalasi share a scale, a vaadi and a
samvaadi, and that Bairagi shares nothing with either. Two ways to tell the network:

    graded      soft targets. Instead of a one-hot, the target puts (1-alpha) on the true
                raag and spreads alpha over its musical neighbours, in proportion to
                affinity**gamma. Cheap, architecture-independent, and it optimises almost
                exactly the `affinity_ce` metric the runs are graded on.
    occupancy   an auxiliary head. The model additionally predicts the true raag's 12-bin
                swar-occupancy vector from the DB. This shares structure across raags --
                every Kafi-thaat raag pulls the representation in a similar direction -- so
                the thin classes (18 clips) borrow from the fat ones (73).

Both are *priors*, which is the shape ../motif-classifier found to work: blending toward the
DB beat both pure learning and pure DB, at every lambda they tried. Neither replaces the
training signal; `alpha=0` / `aux_weight=0` recovers plain cross-entropy exactly.
"""

import torch
import torch.nn.functional as F

from . import dbprior


def graded_targets(alpha=0.3, gamma=4.0, device="cpu"):
    """(50, 50) target rows: (1-alpha) one-hot + alpha * affinity-weighted neighbours."""
    q = torch.from_numpy(dbprior.soft_targets(gamma)).float()
    return ((1.0 - alpha) * torch.eye(q.shape[0]) + alpha * q).to(device)


def occupancy_targets(device="cpu"):
    """(50, 12) row-stochastic swar occupancy, read off the DB."""
    return torch.from_numpy(dbprior.swar_occupancy()).float().to(device)


class Objective:
    """Cross-entropy, optionally graded by the DB, optionally with an auxiliary head.

        loss_fn = Objective(graded_alpha=0.3, aux_weight=0.2)
        loss, parts = loss_fn(outputs, batch)

    `parts` is a dict of the individual terms, logged per epoch so a run that improves only
    because the auxiliary term collapsed is visible rather than mysterious.
    """

    def __init__(self, graded_alpha=0.0, graded_gamma=4.0, aux_weight=0.0, device="cpu"):
        self.graded_alpha = float(graded_alpha)
        self.aux_weight = float(aux_weight)
        self.Q = graded_targets(graded_alpha, graded_gamma, device) if graded_alpha > 0 else None
        self.O = occupancy_targets(device) if aux_weight > 0 else None

    def __call__(self, outputs, batch):
        logits = outputs["logits"]
        y = batch["labels"].to(logits.device)
        if self.Q is None:
            main = F.cross_entropy(logits, y)
        else:
            main = -(self.Q[y] * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
        parts = {"ce": float(main.detach())}

        total = main
        if self.aux_weight > 0:
            if "occupancy" not in outputs:
                raise KeyError("aux_weight > 0 but the model returned no 'occupancy' head")
            # KL(target || predicted) over the 12 swars; target is a proper distribution
            pred = F.log_softmax(outputs["occupancy"], dim=-1)
            aux = -(self.O[y] * pred).sum(dim=-1).mean()
            parts["aux"] = float(aux.detach())
            total = total + self.aux_weight * aux
        return total, parts

    def describe(self):
        bits = ["cross-entropy"]
        if self.graded_alpha:
            bits.append(f"graded(alpha={self.graded_alpha})")
        if self.aux_weight:
            bits.append(f"occupancy-aux(w={self.aux_weight})")
        return " + ".join(bits)
