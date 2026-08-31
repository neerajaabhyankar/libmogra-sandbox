import argparse
import re
import sys
import time

import librosa
import numpy as np
import sounddevice as sd


# PLOT calls plt.show() making a window pop up
PLOT = True

METHODS = ('pyin', 'crepe', 'praat', 'tony')

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


def parse_methods(tokens):
    """Flatten every spelling of --method into one ordered, deduped list.

        --method pyin crepe          space-separated
        --method=pyin --method=crepe repeated flag -- which is what a shell turns
                                     `--method={pyin,crepe}` into, so this form
                                     has to accumulate rather than overwrite
        --method pyin,crepe          comma-separated
    """
    names = [m for token in tokens or [] for m in re.split(r"[,\s]+", token) if m]
    return list(dict.fromkeys(names)) or ["pyin"]


def extract(method, audio, sr, tonic_hz):
    """Run one tracker. Plotting is left to the caller so several can share a figure."""
    if method == 'pyin':
        from trackers.pyin_tracker import extract_relative_pitch_pyin
        return extract_relative_pitch_pyin(audio, sr, plot=False, provided_tonic_hz=tonic_hz)
    elif method == 'crepe':
        from trackers.crepe_tracker import extract_relative_pitch_crepe
        return extract_relative_pitch_crepe(audio, sr, plot=False, provided_tonic_hz=tonic_hz)
    elif method == 'praat':
        from trackers.praat_tracker import extract_relative_pitch_praat
        return extract_relative_pitch_praat(audio, sr, plot=False, provided_tonic_hz=tonic_hz)
    elif method == 'tony':
        from trackers.tony.tony_tracker import extract_relative_pitch_tony
        return extract_relative_pitch_tony(audio, sr, plot=False, provided_tonic_hz=tonic_hz)
    raise ValueError(f"unknown method {method!r}, expected one of {METHODS}")


def as_swaras(notes):
    """Quantized swara names, skipping notes too short to be worth reading out."""
    from visualize import SWARA_LABELS
    return [
        SWARA_LABELS[int(round(n.cents_relative / 100)) % 12]
        for n in notes
        if (n.t_end - n.t_start) >= DURATION_THRESHOLD
    ]


def main(duration, methods=('pyin',)):
    """Two-phase recording: tonic hum first, then melody.

    Every method in `methods` runs on the same recording and they are plotted
    stacked in the given order, on one shared time and pitch axis.
    """

    input("Press Enter and hum your tonic (Sa)...")
    tonic_audio, tonic_sr = record(TONIC_DURATION, label="Please hum the tonic (Sa)")
    tonic_hz = infer_tonic(tonic_audio, tonic_sr)
    note_name = librosa.hz_to_note(tonic_hz)
    print(f"  Inferred tonic: {tonic_hz:.1f} Hz ({note_name})")

    input("\nPress Enter and hum your melody...")
    audio, sr = record(duration, label="Now hum the melody")

    results = []
    for method in methods:
        notes = extract(method, audio, sr, tonic_hz)
        results.append((method, notes))
        label = f"\nMelody ({method}):" if len(methods) > 1 else "\nMelody:"
        print(label, " ".join(as_swaras(notes)))

    if PLOT:
        # Trajectory panels on top, then the histogram pair: the same notes seen as a
        # distribution rather than a sequence, both against the tonic just hummed.
        from freq_histogram import plot_relative_pitch_with_histograms
        plot_relative_pitch_with_histograms(results, tonic_hz=tonic_hz)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract melody from live mic audio.')
    parser.add_argument('--duration', type=float, default=10.0, help='Seconds to record melody.')
    parser.add_argument('--method', type=str, nargs='+', action='extend', default=None,
                        metavar='METHOD',
                        help=f"One or more of: {', '.join(METHODS)}. Plots stack in the "
                             "order given. Space-separated, comma-separated, or the flag "
                             "repeated all work.")
    args = parser.parse_args()

    methods = parse_methods(args.method)
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        parser.error(f"unknown method(s): {', '.join(unknown)}; "
                     f"choose from {', '.join(METHODS)}")

    main(args.duration, methods=methods)
