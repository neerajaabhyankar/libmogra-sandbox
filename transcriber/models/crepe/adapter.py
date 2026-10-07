"""CREPE (torchcrepe, weights ship with the package): a learned monophonic pitch tracker.

Reuses `../raag-identifier/melody-extraction/trackers/crepe_tracker.torchcrepe_predict`
(outside ../raag-identifier/utils: melody-extraction, allowed by CLAUDE.md). It follows the
loudest pitch, so it is the control for "a learned tracker, no lead selection".
"""

import sys

import numpy as np
import torch

from ... import config as C
from ...contract import Track

NAME = "crepe"
P = C.CREPE
SR = P["sr"]


def _predict():
    for d in (C.MELODY_EXTRACTION, C.MELODY_EXTRACTION / "trackers"):
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))
    import crepe_tracker
    crepe_tracker.MODEL_SIZE, crepe_tracker.HOP_LENGTH = P["model"], P["hop"]
    return crepe_tracker.torchcrepe_predict


def transcribe(audio, sr):
    assert sr == SR, f"CREPE expects {SR} Hz"
    wav = torch.from_numpy(np.ascontiguousarray(audio, np.float32)).unsqueeze(0)
    with torch.no_grad():
        f0, conf = _predict()(wav, device=C.DEVICE)
    f0, conf = f0.squeeze(0).cpu().numpy(), conf.squeeze(0).cpu().numpy()
    return Track(np.where(conf >= P["voiced_conf"], f0, 0.0), P["hop"] / SR, conf)
