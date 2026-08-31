# Tony

[Tony](https://www.sonicvisualiser.org/tony/) is a GUI melody-transcription app from the
Sonic Visualiser people. There is no batch/CLI mode and the macOS build (2.1.1) is a
64-bit **Intel** binary, so scripting the app itself is a dead end.

It doesn't matter, because Tony is a front-end. Its transcription engine is the **pYIN
Vamp plugin** (Mauch & Dixon, ICASSP 2014) and the GUI is a hand-correction layer over the
plugin's two outputs. So this module hosts the same plugin in-process via the `vamp`
Python module and reads those outputs directly:

| pYIN output | what Tony shows | what we do with it |
|---|---|---|
| `smoothedpitchtrack` | the blue pitch curve | frame-level `f0_hz` + voiced mask |
| `notes` | the note rectangles | note events, straight from pYIN's note HMM |

## How this differs from `pyin_tracker.py`

`librosa.pyin` implements pYIN's *frame-level* stage only; we then segment notes with our
own `segment_notes()`. Tony's plugin adds a **note-level HMM** on top — onset sensitivity
plus duration pruning — which is the part that isn't in librosa. That is the whole reason
this module exists, so `note_source="tony"` is the default.

```python
from trackers.tony.tony_tracker import extract_relative_pitch_tony

notes = extract_relative_pitch_tony(audio, sr)                          # pYIN note HMM
notes = extract_relative_pitch_tony(audio, sr, note_source="pipeline")  # shared segment_notes()
```

`note_source="pipeline"` routes the frame track through the same `run_pipeline()` the other
three trackers use, so you can compare like for like — only the f0 estimator differs. Tonic
estimation is the shared heuristic in both modes.

Note-duration control differs by mode: `"pipeline"` uses our 0.2 s `min_note_dur` floor,
`"tony"` uses the plugin's own `prunethresh` (default 0.1 s, max 0.2 s):

```python
extract_relative_pitch_tony(audio, sr, parameters={"prunethresh": 0.2})
```

Other pYIN knobs worth touching for gamak-heavy material: `onsetsensitivity` (0.7 default —
lower it to stop slides being chopped into separate notes) and `threshdistr`.

## Install

Two pieces, neither of which touches this repo.

**1. The Python Vamp host.** Build isolation must be off — its `setup.py` imports numpy
directly, so an isolated build env fails with `ModuleNotFoundError: No module named 'numpy'`:

```
pip install --no-build-isolation vamp
```

(Deliberately not added to `pyproject.toml`: `poetry install` always builds isolated and
would hit exactly that failure.)

**2. The pyin plugin binary.**

```
python trackers/tony/install_pyin_plugin.py          # download + install
python trackers/tony/install_pyin_plugin.py --check  # verify
```

The standalone pyin download and Tony's own bundle are Intel-only, but the **Vamp Plugin
Pack 2.0** installer is a universal binary carrying a universal (x86_64 + arm64)
`pyin.dylib` stored uncompressed inside it. The script mounts the dmg and carves that dylib
straight out of the installer binary instead of running the GUI installer, then writes it to
`~/Library/Audio/Plug-Ins/Vamp/pyin.dylib` — the standard per-user Vamp path, no admin
rights, no binary in the repo. Sonic Visualiser and Tony itself will also pick it up there.

If you'd rather not carve: mount [the dmg](https://github.com/vamp-plugins/vamp-plugin-pack/releases/download/v2.0/Vamp.Plugin.Pack.Installer-2.0.dmg)
and run the installer by hand — it puts the same file in the same place, plus ~20 other plugins.

## Alternative route, not taken

[`sonic-annotator`](https://vamp-plugins.org/sonic-annotator/) is the official CLI Vamp host
and would give the same numbers via `sonic-annotator -d vamp:pyin:pyin:notes -w csv`. It
means shelling out and parsing CSV per call, so the in-process `vamp` module is simpler
here — but it's the fallback if the Python host ever stops building.
