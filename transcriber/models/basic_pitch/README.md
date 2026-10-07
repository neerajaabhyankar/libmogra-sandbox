# basic_pitch

**What:** Spotify's Basic Pitch (ICASSP 2022, Apache-2.0), `basic-pitch[onnx]` 0.4.0 via poetry
(macOS-only marker, so the lock does not pull Linux TensorFlow pins). ONNX backend, CPU.

**Outputs used:** of its three maps (note, onset, *contour*), the contour map: salience for 264
pitch bins (3 per semitone from 27.5 Hz) every 256 samples at 22050 Hz (~86 frames/s). Its note
events (integer MIDI + pitch bends) are not used yet.

**Lead selection:** the map is polyphonic with no instrument labels, so `adapter.lead_path` picks
the highest-salience path within 60–1100 Hz that pays a cost per bin of jump (dynamic
programming), refines it between bins with a parabola, and marks frames with salience < 0.3
unvoiced. Settings: `config.BASIC_PITCH`.

**Calibration:** pure tones at 110/220/330/440 Hz peak one bin (33 c) sharp, consistently; Basic
Pitch's own bin-to-pitch mapping does not correct it, so `bin_offset = -1` does (after: 110.1,
220.2, 440.4 Hz).

**Cost:** ~1 s per 30 s of audio; all 85 min of choosing ranges in 75 s.

**First check (one notated 30 s range vs Melodia):** voiced 72% (Melodia 82%); where both are
voiced, median 19 c apart, 62% within 50 c. With `+demucs` separation: 69%, 19 c, 63% -- no change.

**Adaptation:** none yet.
