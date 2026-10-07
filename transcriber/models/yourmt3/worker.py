"""YourMT3+ in its own process: its code has top-level `config`, `model` and `utils` packages that
would collide with the caller's (pakad-matcher's `config`, raag-identifier's `utils`).

Protocol on stdin/stdout: the caller writes an 8-byte little-endian sample count, then that many
float32 samples (16 kHz mono) per request;
the worker answers one JSON line: [[is_drum, program, onset, offset, pitch], ...]. Started by
adapter.py; `python -m transcriber.models.yourmt3.worker <checkpoint key>` (a key of
config.YOURMT3["checkpoints"]) with the vendor dir as working dir.
"""

import importlib.machinery
import json
import sys
import types

import numpy as np
import torch

from transcriber import config as C

P = C.YOURMT3
sys.path[:0] = [str(P["vendor"] / "amt" / "src"), str(P["vendor"])]
_wandb = types.ModuleType("wandb")        # their training logger: only wandb.Table is touched,
_wandb.__spec__ = importlib.machinery.ModuleSpec("wandb", None)   # to build a log table, so a
_wandb.Table = lambda *a, **k: None       # do-nothing placeholder stands in (nothing is logged)
sys.modules.setdefault("wandb", _wandb)

from model_helper import load_model_checkpoint  # noqa: E402
from utils.audio import slice_padded_array  # noqa: E402
from utils.event2note import merge_zipped_note_events_and_ties_to_notes  # noqa: E402
from utils.note2event import mix_notes  # noqa: E402

SR = 16000


def notes(m, audio):
    n_in = m.audio_cfg["input_frames"]
    seg = slice_padded_array(torch.from_numpy(audio)[None], n_in, n_in)
    seg = torch.from_numpy(seg.astype("float32")).to(C.DEVICE).unsqueeze(1)
    with torch.no_grad():
        tokens, _ = m.inference_file(bsz=P["batch"], audio_segments=seg)
    starts = [n_in * i / SR for i in range(seg.shape[0])]
    per_ch = []
    for ch in range(m.task_manager.num_decoding_channels):
        zipped, _, _ = m.task_manager.detokenize_list_batches([a[:, ch, :] for a in tokens], starts,
                                                              return_events=True)
        per_ch.append(merge_zipped_note_events_and_ties_to_notes(zipped)[0])
    return mix_notes(per_ch)


def main():
    out = sys.stdout
    sys.stdout = sys.stderr                     # their prints must not corrupt the protocol
    m = load_model_checkpoint(args=list(P["checkpoints"][sys.argv[1]]), device="cpu").to(C.DEVICE)
    out.write("ready\n"); out.flush()
    stdin = sys.stdin.buffer
    while len(head := stdin.read(8)) == 8:
        audio = np.frombuffer(stdin.read(4 * int.from_bytes(head, "little")), np.float32).copy()
        ns = notes(m, audio)
        out.write(json.dumps([[n.is_drum, n.program, n.onset, n.offset, n.pitch] for n in ns]) + "\n")
        out.flush()


if __name__ == "__main__":
    main()
