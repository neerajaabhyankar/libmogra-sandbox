# raagdataset-tonics

Hand-annotated tonic (Sa) for every video in `../hindustani-raag-small`.

**Nothing here modifies `../raagdataset.ipynb`.** That notebook built the dataset; this is a
separate pass that adds one field to it.

## Why

`../motif-classifier/ceilings.py` measures what a correct tonic is worth. Swapping the
estimated tonic for an oracle one takes 50-way top-1 from **0.109 to 0.314** and top-5 from
**0.277 to 0.721** — about 3× the value of every other modelling choice in that folder
combined. Every automatic estimator tried there (mode-of-pitch, a sub-semitone tuning
refinement, a Sa-vs-Pa chroma template, marginalising over all 12 rotations) left most of
that gap on the table. So: label it.

426 videos, 50 raags. One tonic per **video**, not per clip — the three chunks of a video
come from one recording and share a Sa.

## Run it

```bash
poetry run python annotate_tonics.py                # everything not yet done
poetry run python annotate_tonics.py --limit 25     # a 20-minute session
poetry run python annotate_tonics.py --raag Yaman   # one raag at a time
poetry run python annotate_tonics.py --status       # progress, and which rows need review
poetry run python annotate_tonics.py --review       # replay annotations with a sine at the stored Sa
poetry run python annotate_tonics.py --redo --video ABC123    # fix one
```

Per video: a 10 s excerpt from the middle of a clip plays, the YouTube URL is printed in
case you want more context, you hum Sa for 4 s, and the result is saved immediately.

At every prompt: `Enter` accept · `r` replay · `l` longer excerpt · `p` play your Sa as a
sine mixed against the clip · `a` re-hum · `s` skip · `q` quit.

**Resumable by construction.** `tonics.csv` is appended and flushed after each annotation
and re-read at startup; annotated videos are skipped. Ctrl-C is safe.

## The snapping step

A hummed Sa is worth maybe ±20 cents. The recording states its own Sa far more precisely.
So the hum is not used as the label — it is used to *choose* which peak of the video's own
pitch histogram is Sa, and that peak becomes the label:

1. pYIN (via `../melody-extraction`) over up to 3 chunks of the video → f0 frames
2. histogram in **10-cent bins** (deliberately finer than a semitone, and not on an A440
   grid — the label should be where the tanpura actually is)
3. light smoothing so vibrato doesn't split a note into two peaks
4. nearest peak within `--snap-cents` (default 60), also checking ±1 and ±2 octaves so an
   octave-displaced hum still lands correctly
5. the peak position is refined by parabolic interpolation, so the label is not pinned to
   the 10-cent bin grid
6. the result is folded into a standard octave band (95–260 Hz). That band spans more than
   an octave on purpose — a male and a female Sa genuinely differ by that much — so when two
   octave candidates both land in range, whichever carries more histogram mass wins
7. no peak in range → the raw hum is kept and the row is flagged `snapped=NO`

Octave handling is the part most likely to produce silent 2× errors in a label file, so it
is handled in two places: step 4 prefers the octave you actually hummed (only falling back
to ±1/±2 octaves if nothing there is close enough), and step 6 makes the stored value
octave-consistent across the dataset. Checked against real clips: four simulated hums of the
same Sa — 22 cents flat, 35 cents sharp, an octave down, an octave up — converge to within
~5 cents on the same label.

`--status` lists the flagged rows and `--review` replays them, so those can be revisited.

## Output

`tonics.csv`, one row per video:

| column | meaning |
|---|---|
| `video` | YouTube id, as it appears in the clip filenames |
| `raag` | dataset folder name |
| `tonic_hz` | **the label** — snapped Sa in Hz |
| `note` | nearest named pitch, for eyeballing |
| `hum_hz` | what you actually hummed, kept for audit |
| `snap_cents` | how far the hum was moved |
| `snapped` | `yes`, or `NO` if the raw hum was kept |
| `clip` | which chunk you heard |
| `timestamp` | when |

Not pushed anywhere. Once the file is complete we can decide how it joins the Hub dataset.

## Consuming it

```python
import csv
tonics = {r["video"]: float(r["tonic_hz"]) for r in csv.DictReader(open("tonics.csv"))}
```

In `../motif-classifier`, `represent.Params` already takes a per-video tonic — wiring this
in means replacing `_video_tonics()` with a lookup, and re-running
`tune.py` / `report.py` to see how much of the 0.109 → 0.314 oracle gap is real rather than
an artefact of the oracle being allowed to choose per clip.

## Requirements

`sounddevice` (already in the project's `pyproject.toml`) plus a working mic and speakers.
The pYIN Vamp plugin is needed for the snapping histogram — see
`../melody-extraction/trackers/tony/README.md` if it is not installed.
