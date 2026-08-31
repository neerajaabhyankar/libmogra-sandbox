# Shared helpers for the embedding-probe scripts (01_probe_embeddings.py,
# 02_sweep_head.py). See plan.md for context.

import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

EMB_DIR = Path(__file__).parent.parent / "embeddings-exploration" / "outputs" / "2s" / "crc-jeevster"
DATASET_ID = "neerajaabhyankar/hindustani-raag-small"
DATASET_REVISION = "0dfb021e54e0e7489b90a47e23ef15f34fa740ec"
LABEL_NAMES = ["AheerBhairav", "AlhaiyaBilawal", "Bageshree", "Bahar", "Bairagi"]


def load_split(split: str):
    files = sorted(
        EMB_DIR.glob(f"{split}_*.npz"),
        key=lambda p: int(re.search(r"_(\d+)\.npz", p.name).group(1)),
    )
    idxs = [int(re.search(r"_(\d+)\.npz", f.name).group(1)) for f in files]

    from datasets import load_dataset

    ds = load_dataset(DATASET_ID, revision=DATASET_REVISION)[split]

    X, y = [], []
    for f, idx in zip(files, idxs):
        X.append(np.load(f)["clip_mean"])
        y.append(ds[idx]["label"])
    return np.stack(X), np.array(y)


class MLPHead(nn.Module):
    """Linear probe (hidden_dims=()) or MLP with 1-2 hidden layers."""

    def __init__(self, in_dim, num_classes, hidden_dims=(), batchnorm=False, dropout=0.0):
        super().__init__()
        layers = []
        dims = [in_dim, *hidden_dims]
        for i in range(len(hidden_dims)):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if batchnorm:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
