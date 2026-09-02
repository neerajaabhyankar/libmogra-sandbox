"""Identify the raag in an audio file.

    python predict.py alap.wav --tonic-hz 146.8
    python predict.py alap.wav --tonic-file my_sa.wav      # 5 s of a held Sa
    python predict.py alap.wav --tonic-note D3

The tonic is required. A raag is a pattern of intervals above Sa, so a model given the
wrong Sa is not slightly wrong -- it is answering a different question.
"""

import argparse

from raag_fusion import RaagIdentifier, audio, tonic


def resolve_tonic(args):
    if args.tonic_hz:
        return float(args.tonic_hz), "given"
    if args.tonic_note:
        import librosa

        return float(librosa.note_to_hz(args.tonic_note)), f"from {args.tonic_note}"
    y, sr = audio.load(args.tonic_file)
    return tonic.from_hum(y, sr), f"heard in {args.tonic_file}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", help="the recording to identify (20 s or more works best)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--tonic-hz", type=float, help="Sa in Hz")
    g.add_argument("--tonic-note", help="Sa as a note name, e.g. C#3")
    g.add_argument("--tonic-file", help="a few seconds of a held Sa")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    tonic_hz, how = resolve_tonic(a)
    y, sr = audio.load(a.audio)
    print(f"{a.audio}: {len(y) / sr:.0f}s, Sa = {tonic_hz:.1f} Hz ({how})")

    model = RaagIdentifier.load(device=a.device)
    for i, p in enumerate(model.predict(y, sr, tonic_hz, top_k=a.top_k), 1):
        print(f"  {i}. {p}")


if __name__ == "__main__":
    main()
