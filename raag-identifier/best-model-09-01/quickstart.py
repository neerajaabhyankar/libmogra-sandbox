"""Hum into your microphone, get five guesses back.

    python quickstart.py                     # press Enter to record Sa, then to sing
    python quickstart.py --tonic-hz 146.8    # skip the Sa recording
    python quickstart.py --max-seconds 90

Needs `sounddevice` on top of the base requirements:

    pip install sounddevice

What to sing: hold Sa steadily for the first recording -- one note, no scoops. Then sing an
alap or a bandish in that raag, for at least 20 seconds and ideally longer. Sing *in the
raag*, not a scale: the model reads which swars you dwell on and how often, and a plain
aaroha-avaroha tells it much less than real phrases do.
"""

import argparse
import sys
import threading
import time

import numpy as np

from raag_fusion import RaagIdentifier, tonic

SR = 22050
BAR_WIDTH = 36


def _bar(elapsed, limit, tail=""):
    filled = int(BAR_WIDTH * min(elapsed / limit, 1.0))
    sys.stdout.write(f"\r  [{'█' * filled}{'░' * (BAR_WIDTH - filled)}] {elapsed:5.1f}s {tail}")
    sys.stdout.flush()


def record(seconds, stop_on_enter=False):
    """Record up to `seconds`, optionally stopping early when the user presses Enter."""
    import sounddevice as sd

    stop = threading.Event()
    if stop_on_enter:
        # a daemon thread, so a recording that runs to the time limit does not leave the
        # program waiting on an Enter that is never coming
        threading.Thread(target=lambda: (sys.stdin.readline(), stop.set()),
                         daemon=True).start()

    frames = []
    with sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                        callback=lambda data, *_: frames.append(data.copy())):
        start = time.time()
        while not stop.is_set() and (elapsed := time.time() - start) < seconds:
            _bar(elapsed, seconds, "(Enter to stop)" if stop_on_enter else "")
            time.sleep(0.05)
        elapsed = time.time() - start
    _bar(elapsed, seconds, "done            ")
    print()
    return np.concatenate(frames).reshape(-1) if frames else np.zeros(0, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-seconds", type=float, default=60.0,
                    help="the singing stops here by itself (default 60)")
    ap.add_argument("--tonic-seconds", type=float, default=5.0)
    ap.add_argument("--tonic-hz", type=float, default=None, help="skip recording Sa")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    print("Loading the model...")
    model = RaagIdentifier.load(device=a.device)

    tonic_hz = a.tonic_hz
    if tonic_hz is None:
        input(f"\nPress Enter, then hold your Sa for {a.tonic_seconds:.0f}s ")
        y = record(a.tonic_seconds)
        tonic_hz = tonic.from_hum(y, SR)
    print(f"Sa = {tonic_hz:.1f} Hz")

    input(f"\nPress Enter and sing the raag (minimum 20s, maximum {a.max_seconds:.0f}s). Enter again to stop, or it ends at "
          f"{a.max_seconds:.0f}s ")
    y = record(a.max_seconds, stop_on_enter=True)

    if float(np.max(np.abs(y), initial=0.0)) < 1e-3:
        sys.exit("that recording is silent -- check your input device")
    if len(y) / SR < 20:
        print(f"  note: {len(y) / SR:.0f}s is short. The model was trained on 20s windows "
              f"and pads anything shorter, so expect worse guesses.")

    print("\nAnalyzing...\n")
    for i, p in enumerate(model.predict(y, SR, tonic_hz, top_k=a.top_k), 1):
        print(f"  {i}. {p.raag:<20s} {p.probability:6.1%}  "
              f"{'█' * int(round(30 * p.probability))}")
    print("\nThe right raag is the top one < half the time and in this top-5 list about "
          "80% of the time. Trained on YouTube recordings of 50 raags. Take this with a bag of salt!")


if __name__ == "__main__":
    main()
