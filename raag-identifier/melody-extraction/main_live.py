import argparse
import sys
import time

import librosa
import numpy as np
import sounddevice as sd


# PLOT calls plt.show() making a window pop up
PLOT = True

SAMPLE_RATE = 22050
TONIC_DURATION = 4.0  # seconds to record for tonic estimation
DURATION_THRESHOLD = 0.2  # notes shorter than this (seconds) are excluded from printout


def record(duration, sr=SAMPLE_RATE, label="Recording"):
    print(f"\n{label} — {duration:.1f}s")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")

    bar_width = 40
    start = time.time()
    while True:
        elapsed = time.time() - start
        if elapsed >= duration:
            break
        frac = min(elapsed / duration, 1.0)
        filled = int(bar_width * frac)
        bar = "█" * filled + "░" * (bar_width - filled)
        sys.stdout.write(f"\r  [{bar}] {elapsed:.1f}s / {duration:.1f}s")
        sys.stdout.flush()
        time.sleep(0.05)

    sys.stdout.write(f"\r  [{'█' * bar_width}] {duration:.1f}s / {duration:.1f}s\n")
    sys.stdout.flush()
    sd.wait()
    return audio.reshape(-1), sr


def infer_tonic(audio, sr):
    """Run pyin on a short clip and return tonic in Hz."""
    from pipeline import estimate_tonic_hz

    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

    fmin = librosa.note_to_hz("C2")
    fmax = librosa.note_to_hz("C7")
    frame_length = 2048
    hop_length = frame_length // 4

    f0_hz, voiced_flag, _ = librosa.pyin(
        audio,
        fmin=fmin,
        fmax=fmax,
        sr=SAMPLE_RATE,
        frame_length=frame_length,
        hop_length=hop_length,
        fill_na=0.0,
    )
    voiced_mask = voiced_flag & (f0_hz > 0)
    return estimate_tonic_hz(f0_hz, voiced_mask)


def main(duration, method='pyin'):
    """Two-phase recording: tonic hum first, then melody."""

    input("Press Enter and hum your tonic (Sa)...")
    tonic_audio, tonic_sr = record(TONIC_DURATION, label="Please hum the tonic (Sa)")
    tonic_hz = infer_tonic(tonic_audio, tonic_sr)
    note_name = librosa.hz_to_note(tonic_hz)
    print(f"  Inferred tonic: {tonic_hz:.1f} Hz ({note_name})")

    input("\nPress Enter and hum your melody...")
    audio, sr = record(duration, label="Now hum the melody")

    if method == 'pyin':
        from trackers.pyin_tracker import extract_relative_pitch_pyin
        notes = extract_relative_pitch_pyin(audio, sr, plot=PLOT, provided_tonic_hz=tonic_hz)
    elif method == 'crepe':
        from trackers.crepe_tracker import extract_relative_pitch_crepe
        notes = extract_relative_pitch_crepe(audio, sr, plot=PLOT, provided_tonic_hz=tonic_hz)
    elif method == 'praat':
        from trackers.praat_tracker import extract_relative_pitch_praat
        notes = extract_relative_pitch_praat(audio, sr, plot=PLOT, provided_tonic_hz=tonic_hz)

    from visualize import SWARA_LABELS
    melody = [
        SWARA_LABELS[int(round(n.cents_relative / 100)) % 12]
        for n in notes
        if (n.t_end - n.t_start) >= DURATION_THRESHOLD
    ]
    print("\nMelody:", " ".join(melody))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract melody from live mic audio.')
    parser.add_argument('--duration', type=float, default=10.0, help='Seconds to record melody.')
    parser.add_argument('--method', type=str, default='pyin', choices=['pyin', 'crepe', 'praat'])
    args = parser.parse_args()

    main(args.duration, method=args.method)
