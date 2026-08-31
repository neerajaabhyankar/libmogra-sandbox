# source-separation — get the melody out, cheaply

A reusable module, not an experiment. Anything in this repo that needs "the main melodic
line, without the tabla" imports it:

```python
import sys; sys.path.insert(0, ".../raag-identifier/source-separation")
from separation import separate, available

stems = separate(audio, sr, backend="hpss")
stems.melody          # np.ndarray at the same sr — the only field callers need
stems.percussion      # tabla, where the backend produces it
stems.energy_split()  # {"melody": 0.60, "percussion": 0.31, ...}
```

The design goal is **cheap and good enough**, not clean stems. Some tabla bleeding into the
melody is fine; what is not fine is a pitch tracker chasing a tabla stroke or sitting on the
tanpura for half a clip.

## Backends

| backend | what it does | cost | new deps |
|---|---|---|---|
| `none` | pass the mixture through — the control | free | — |
| `hpss` | librosa harmonic/percussive median filtering | ~0.5 s / 20 s clip | none |
| `hpss+drone` | `hpss`, then a stationarity notch for the tanpura | ~1 s | none |
| `demucs` | HT-Demucs `htdemucs`, 4-stem; `vocals` is the melody | ~4 s (MPS) | `pip install demucs` |
| `demucs+drone` | as above plus the notch | ~4 s | `pip install demucs` |

`hpss` needs no model and no download: tabla is broadband and vertical in a spectrogram,
sustained melody is narrowband and horizontal, and median-filtering along each axis
separates them. `margin=3` makes the split conservative — energy that is not clearly one or
the other is dropped from both, which is the right bias here.

**Not implemented, deliberately:** BS-RoFormer / Mel-Band RoFormer fine-tuned on Saraga
multitracks. That is the actually-good answer for this repertoire and a research project of
its own; `BACKENDS` is the seam it slots into. The reference notes MTG's Carnatic ISMIR23
separation model as the nearest in-domain pretrained option — Carnatic, not Hindustani.

## Inspecting it on its own

```bash
poetry run python inspect_separation.py --list                 # what can run here
poetry run python inspect_separation.py --audio FILE --out /tmp/sep --plot
poetry run python inspect_separation.py --clip Yaman --n 3 --out /tmp/sep
```

That writes `mixture.wav` plus `melody/percussion/drone/residual.wav` per backend, so the
first check is **listen to them**. It also prints proxy metrics, because there is no
ground-truth stem here and so no SDR to compute:

| metric | reads on | direction |
|---|---|---|
| `voiced%` | fraction of frames CREPE is confident about | up is better |
| `conf` | mean CREPE confidence | up |
| `jitter¢` | median cents change between adjacent voiced frames | down |
| `%Sa` | share of voiced frames within 25c of the tonic's pitch class | **flat** |
| `H(pitch)` | entropy of the octave-folded 12-bin histogram | down |
| `melody energy` | share of input energy kept in the melody stem | context only |

**`%Sa` is the guard, and it is not optional.** A tracker that has locked onto the tanpura
produces beautiful numbers everywhere else — voiced% up, jitter near zero — and is useless.
Only `%Sa` catches it. Never read `jitter` without it.

## What the metrics said

Two Yaman clips, all backends (`inspect_separation.py --clip Yaman --n 2`):

| backend | voiced% | conf | jitter¢ | H(pitch) | melody energy |
|---|---|---|---|---|---|
| none | 83.7 / 82.4 | 0.62 / 0.64 | 18.3 / 17.3 | 3.30 / 3.04 | 1.00 |
| **hpss** | **88.5 / 90.8** | 0.66 / 0.70 | **2.2 / 1.1** | **3.10 / 2.72** | 0.60 / 0.57 |
| hpss+drone | 90.5 / 91.6 | 0.69 / 0.72 | 1.2 / 0.9 | 3.23 / 3.22 | 0.41 / 0.35 |
| demucs | 86.2 / 88.8 | 0.68 / 0.72 | 18.7 / 17.9 | 3.33 / 3.23 | 0.79 / 0.72 |

1. **HPSS is the winner on every axis that matters** — more voiced frames, higher
   confidence, an order of magnitude less jitter, and a peakier pitch histogram.
2. **The jitter collapse (18 → 1 cent) was checked for drone-locking and is real.** Across
   four Yaman/Malkauns clips, `%Sa` stayed flat under HPSS (20.1→22.0, 5.9→6.0, 35.2→36.1,
   39.0→43.3) and the count of occupied pitch bins did not collapse. The tracker is
   following the melody, just a cleaner version of it.
3. **Drone suppression hurts the histogram.** `hpss+drone` has the best voiced% and jitter
   but its entropy goes back *up* (3.10→3.23, 2.72→3.22). The notch is removing sustained
   Sa, which is real melodic content — a long nyas on Sa looks exactly like a drone to a
   stationarity test. Use `hpss+drone` if you want a clean f0 contour; use plain `hpss` if
   anything downstream cares about the pitch distribution.
4. **Demucs does not help here, and sometimes hurts.** Jitter is unchanged from no
   separation at all (18.7 vs 18.3), entropy slightly *worse*, and on instrumental clips it
   cuts `%Sa` hard (35.2→28.7, 39.0→19.0) while dropping voiced% — its `vocals` stem is the
   wrong target when the "melody" is a sitar or sarangi, and it has never heard a tanpura.
   This matches the reference's own warning that Western-trained separators leak pitched
   tabla and do not generalise to this repertoire. **It is 8x slower than HPSS for worse
   results**, so `hpss` is the default.

## Does it help downstream? No — and that is the important result

`../motif-classifier/separation_effect.py` re-extracted CREPE over HPSS-separated audio and
re-ran the strong methods. Every method that sees the separated audio gets slightly *worse*
(M9 0.422 -> 0.393, M12 0.405 -> 0.393, M14 0.464 -> 0.438), while a control method reading
un-re-extracted Tony notes stays put at 0.304 to three decimals.

So the tracker metrics above are real but **do not translate into accuracy**. The likely
reason is visible in those very metrics: jitter falling from ~18 cents to ~1 is partly tabla
bleed being removed and partly **meend and gamak being smoothed away** — and the downstream
methods (a 120-bin pitch histogram, a time-delayed melody surface) exist precisely to read
sub-semitone movement.

What looks like noise to a pitch tracker is partly the signal for raag identification. That
is an argument for a separator that removes percussion *without* touching the melodic line —
a fine-tuned BS-RoFormer — not for tuning HPSS harder.

## Layout

```
source-separation/
  plan.md                 # this file
  separation.py           # Stems, separate(), BACKENDS, available(), suppress_drone()
  inspect_separation.py   # CLI: stems to wav, proxy metrics, spectrogram/f0 plots
```

Consumed by `../motif-classifier/extract.py --separate <backend>`, which caches the
resulting note/f0 tracks to `cache/notes_<tracker>_<version>_<backend>.npz`.
