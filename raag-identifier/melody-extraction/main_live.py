import argparse
import sounddevice as sd


# USE_CHROMA folds to a single octave with swara labels (S r R g G m M P d D n N)
# on the y-axis; set False for the unfolded cents-vs-tonic view (shows octave jumps)
USE_CHROMA = True

# PLOT calls plt.show() making a window pop up
PLOT = True

SAMPLE_RATE = 22050


def record(duration, sr=SAMPLE_RATE):
    print(f"Recording for {duration:.1f}s...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    print("Done recording.")
    return audio.reshape(-1), sr


def main(duration, method='pyin'):
    """
    `notes` contains the note list (t_start, t_end, cents_relative, chroma_cents).
    """

    audio, sr = record(duration)
    if method == 'pyin':
      from trackers.pyin_tracker import extract_relative_pitch_pyin
      notes = extract_relative_pitch_pyin(audio, sr, plot=PLOT, use_chroma=USE_CHROMA)
    elif method == 'crepe':
      from trackers.crepe_tracker import extract_relative_pitch_crepe
      notes = extract_relative_pitch_crepe(audio, sr, plot=PLOT, use_chroma=USE_CHROMA)
    elif method == 'praat':
      from trackers.praat_tracker import extract_relative_pitch_praat
      notes = extract_relative_pitch_praat(audio, sr, plot=PLOT, use_chroma=USE_CHROMA)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract melody from live mic audio.')
    parser.add_argument('--duration', type=float, default=10.0, help='Seconds to record.')
    parser.add_argument('--method', type=str, default='pyin', choices=['pyin', 'crepe', 'praat'])
    args = parser.parse_args()

    main(args.duration, method=args.method)
