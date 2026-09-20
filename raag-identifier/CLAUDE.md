This folder is a dumping ground for all things related to raag identification.
It has datasets, folders (that correspond to different exploration strategies), and within those folders -- models.

## The pitch pathway convention

**Every model version here exposes its pitch track as a standalone module.** The
classifier is one consumer of it, not its owner.

The reason is that the pitch track answers more questions than "which raag". A swar
histogram drawn for a listener, the swars present and occasional, nyas, saptak
distribution, aaroha/avaroha asymmetry — all of them read the same f0 track, and the mogra
app computes it once and shares it. A model that buries the tracker inside its melody
branch forces every one of those to run it again, and to guess at the settings.

So a model package `<version>/<package>/` provides:

| | |
|---|---|
| `pitch.PitchTrack` | `f0_hz`, `voiced`, `hop_seconds`; plus `seconds`, `voiced_fraction`, `cents_above(tonic_hz)` |
| `pitch.track(y, ...)` | audio at `pitch.SR` -> a `PitchTrack` |
| `pitch.from_audio(y, sr, ...)` | audio at any rate -> a `PitchTrack` |
| `melody_branch.histogram(f0, voiced, tonic)` | a distribution over pitch classes summing to 1 |
| `melody_branch.profile(track, tonic)` | the same, from a `PitchTrack` |
| `melody_branch.features(hist)` | whatever compression the classifier was fitted on |

Three properties matter more than the names:

1. **The track is tonic-free.** It is absolute Hz, computable before anyone has decided
   where Sa is. Only the folding needs a tonic.
2. **The histogram is a distribution over time, not a feature vector.** Each bin is the
   share of the voiced passage spent at that pitch class, so it can be drawn as "share of
   your time" with nothing further done to it. Any compression the model wants lives in
   `features`. `best-model-09-01` raises to the power 0.5; drawing that as time would
   misreport every proportion by a square root.
3. **A pitch track is reproducible.** Same recording, same answer, every process. The
   current tracker (Essentia's Melodia) gives this for free. torchcrepe, which
   `best-model-09-01` used until 2026-09-19, did not — it dithered its bins from numpy's
   *global* RNG, so that version pinned the seed and restored the caller's RNG state
   around every call. A tracker that reintroduces randomness has to pin it the same way.

`model_contract.py` checks all of this. Run it against a model directory before calling
that version done:

    python3 model_contract.py best-model-09-01 --package raag_fusion

New versions should start by copying `best-model-09-01/raag_fusion/pitch.py`, which exists
to be copied.

## Other standing rules

- The **datasets** are the confusing part: `hindustani-raag-small` is v0 and is missing 64
  of the 150 test clips. `hindustani-raag-small-v1` is what the published numbers were
  measured on. Check which one a script points at before believing its output.
- Model directories are **uploadable to the Hub** (`upload_to_hub.py`). Anything added to
  one has to work for someone who downloads it and has none of this folder — so a model's
  own `tests/` must not import from this level.
- **Nobody but Neeraja pushes to the Hub.** Prepare the directory, validate it, and ask.
