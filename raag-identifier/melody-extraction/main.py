import argparse
import librosa


# USE_CHROMA folds to a single octave with swara labels (S r R g G m M P d D n N)
# on the y-axis; set False for the unfolded cents-vs-tonic view (shows octave jumps)
USE_CHROMA = True

# PLOT calls plt.show() making a window pop up
PLOT = True


def main(path_to_audio, method='pyin'):
    """
    `notes` contains the note list (t_start, t_end, cents_relative, chroma_cents).
    """
    
    audio, sr = librosa.load(path_to_audio, sr=None, mono=True)
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
    parser = argparse.ArgumentParser(description='Extract melody from audio.')
    parser.add_argument('audio_path', type=str, help='Path to the audio file.')
    parser.add_argument('--method', type=str, default='pyin', choices=['pyin', 'crepe', 'praat'])
    args = parser.parse_args()
    
    main(args.audio_path, method=args.method)
