# ymt3plus -- the MT3 stand-in

**What:** "YMT3+" from the YourMT3 space (518 MB checkpoint in `../yourmt3/vendor/`, GPL-3.0):
MT3's architecture (T5 encoder-decoder, MT3-style tokens), trained by the YourMT3 authors on
data that includes singing voice. Neeraja (2026-10-07): use it instead of Google's MT3, which
had no singing in training (and would re-pin the shared env's TensorFlow/numpy).

**Outputs, lead selection, worker:** as `../yourmt3/README.md` -- same adapter, other checkpoint.

**Adaptation:** none yet.
