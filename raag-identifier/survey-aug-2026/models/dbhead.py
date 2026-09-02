"""A classifier head that scores against the libmogra templates instead of a free matrix.

This is M12 from ../motif-classifier -- "a pitch profile, compared to each raag's expected
pitch profile by chi-square, where the expected profile is the database's blended with a
learned one" -- with the hand-built histogram replaced by a learned front end, and the whole
thing trained end to end.

The reason to want it: an ordinary `Linear(D, 50)` head has 50 independent weight vectors
and must learn each raag from its own 18-73 clips. This head has **one** shared map from
features to a 12-bin swar profile, plus 50 templates that start at what the database already
says. A raag with 18 clips inherits a usable template on epoch zero and only has to learn a
correction to it.

The blend is the same knob M12 tuned, and the same result is expected:

    lam = 0    templates free, database ignored          (M12 at lam=0 -> 0.382)
    lam = 0.3  blended                                   (M12's optimum -> 0.405)
    lam = 1    templates frozen at the database          (M12 at lam=1 -> 0.217)

`learn_templates=False` with `lam=1` is the strong test: the *only* raag-specific parameters
left are 50 scalar biases, and everything else the model knows about which raag is which
comes out of the database.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import dbprior

EPS = 1e-8


class DBTemplateHead(nn.Module):
    """(B, D) features -> (B, 50) logits, via a 12-bin swar profile and chi-square.

    lam            0..1, how far the reference templates are pulled toward the database
    learn_templates  whether the learned half of the blend is trainable at all
    """

    def __init__(self, feature_dim, n_bins=12, lam=0.3, learn_templates=True,
                 hidden=(128,), dropout=0.2):
        super().__init__()
        from .heads import mlp_head

        self.n_bins = n_bins
        self.lam = float(lam)
        self.to_profile = mlp_head(feature_dim, n_bins, hidden, dropout)

        db = dbprior.swar_occupancy() if n_bins == 12 else dbprior.pitch_template(n_bins)
        db = torch.from_numpy(np.asarray(db, dtype=np.float32))
        self.register_buffer("db_templates", db)
        # the learned half starts *at* the database, so epoch 0 is exactly the M12 prior
        self.learned_logits = nn.Parameter(torch.log(db + EPS), requires_grad=learn_templates)
        # chi-square is a distance, not a logit; these turn it into one
        self.scale = nn.Parameter(torch.tensor(10.0))
        self.bias = nn.Parameter(torch.zeros(db.shape[0]))

    def templates(self):
        learned = F.softmax(self.learned_logits, dim=-1)
        return (1.0 - self.lam) * learned + self.lam * self.db_templates

    def forward(self, features):
        p = F.softmax(self.to_profile(features), dim=-1)          # (B, n_bins)
        refs = self.templates()                                    # (50, n_bins)
        chi2 = 0.5 * (((p[:, None, :] - refs[None]) ** 2)
                      / (p[:, None, :] + refs[None] + EPS)).sum(-1)
        return -self.scale.abs() * chi2 + self.bias, p


class RaagClassifierDB(nn.Module):
    """The same wrapper as `heads.RaagClassifier`, but with `DBTemplateHead` on the end.

    Returns the predicted swar profile as `occupancy`, so the auxiliary objective in
    `common.losses` can supervise it directly -- which is the natural pairing: the head
    *interprets* that vector as a pitch profile, so telling it what the profile should be is
    not an extra task, it is the same task stated twice.
    """

    def __init__(self, backbone, feature_dim, num_labels=50, tonic_mode="none",
                 lam=0.3, n_bins=12, learn_templates=True, dropout=0.2, side_dim=0,
                 side_out=64):
        super().__init__()
        from .heads import FiLM, SideFeatures

        if num_labels != 50:
            raise ValueError("the DB templates are the 50 dataset raags")
        self.backbone = backbone
        self.tonic_mode = tonic_mode
        self.film = FiLM(feature_dim) if tonic_mode == "condition" else None
        self.side = SideFeatures(side_dim, side_out) if side_dim else None
        self.head = DBTemplateHead(feature_dim + (self.side.out_dim if self.side else 0),
                                   n_bins=n_bins, lam=lam,
                                   learn_templates=learn_templates, dropout=dropout)

    def forward(self, input_values, tonic=None, side=None):
        from .heads import concat_side

        h = self.backbone(input_values)
        if self.film is not None:
            h = self.film(h, tonic)
        h = concat_side(self.side, h, side)
        logits, profile = self.head(h)
        return {"logits": logits, "features": h, "occupancy": torch.log(profile + EPS)}
