"""YMT3+ -- YourMT3's reproduction of MT3's architecture (a T5 encoder-decoder over spectrograms),
trained on their data, which includes singing voice. Stands in for Google's MT3, which had no
singing in training and does not install in the shared env (plan.md). Same adapter and worker
as yourmt3; only the checkpoint differs (config.YOURMT3["checkpoints"]["ymt3plus"])."""

from functools import partial

from ..yourmt3 import adapter as _ymt3

NAME = "ymt3plus"
SR = _ymt3.SR
transcribe = partial(_ymt3.transcribe, checkpoint=NAME)
