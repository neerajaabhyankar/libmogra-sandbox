"""Look at what separation actually did — audibly, visually, and numerically.

There is no ground-truth stem for these recordings, so "did it work?" cannot be answered
by SDR. What can be answered is the question we actually care about: **is the pitch track
downstream cleaner?** Three proxies, none needing a reference:

  voiced%      fraction of frames CREPE is confident about. Separation that strips the
               tabla should raise this; separation that eats the melody drops it.
  jitter       median |cents| change between adjacent voiced frames. Tabla and drone drag
               a tracker back and forth, so lower is better — but a monotone drone alone
               would score 0, which is why it is never read without voiced% beside it.
  %Sa          share of voiced frames within 25 cents of the tonic's pitch class. This is
               the guard on `jitter`: a tracker that has locked onto the tanpura instead of
               the melody produces beautifully low jitter and a %Sa that shoots up. Read
               the two together, always.
  H(pitch)     entropy of the octave-folded pitch histogram, in bits, over 12 semitone
               bins. A clean melody spends its time on the raag's 5-7 swars, so the
               histogram should get *peakier* (lower entropy) as accompaniment is removed.
               This is the metric closest to what the histogram-fingerprint baseline uses.

Usage — one file, every backend, stems written out so you can listen:

    poetry run python inspect_separation.py --audio path/to.mp3 --out /tmp/sep
    poetry run python inspect_separation.py --clip Yaman --n 3          # from the dataset
    poetry run python inspect_separation.py --audio x.mp3 --plot        # + spectrograms

The wavs it writes are the point: play `melody.wav` and decide for yourself. The numbers
are a triage tool, not a verdict.
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MELODY_DIR = HERE.parent / "melody-extraction"
for p in (str(HERE), str(MELODY_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from separation import separate, available, BACKENDS  # noqa: E402

CONF_THRESHOLD = 0.5


def track_pitch(audio, sr):
    """CREPE f0 + confidence, reusing melody-extraction's own settings."""
    import torch
    import librosa
    from trackers.crepe_tracker import (
        torchcrepe_predict, TARGET_SR, HOP_LENGTH, CONFIDENCE_THRESHOLD,
    )

    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    wav = torch.from_numpy(np.ascontiguousarray(audio)).float().unsqueeze(0)
    with torch.no_grad():
        f0, conf = torchcrepe_predict(wav, device="mps" if torch.backends.mps.is_available() else "cpu")
    return (f0.squeeze(0).numpy(), conf.squeeze(0).numpy(),
            HOP_LENGTH / TARGET_SR, CONFIDENCE_THRESHOLD)


def pitch_metrics(audio, sr, tonic_hz=None):
    f0, conf, hop, thr = track_pitch(audio, sr)
    voiced = conf >= thr
    frac = float(voiced.mean())
    if voiced.sum() < 10:
        return {"voiced": frac, "conf": float(conf.mean()), "jitter": float("nan"),
                "entropy": float("nan"), "near_sa": float("nan")}
    ref = tonic_hz or float(np.median(f0[voiced]))
    cents = 1200.0 * np.log2(np.clip(f0, 1e-9, None) / ref)
    c = cents[voiced]
    jitter = float(np.median(np.abs(np.diff(c)))) if len(c) > 1 else float("nan")

    # octave-folded 12-bin histogram, the same view the fingerprint baseline classifies on
    from freq_histogram import pitch_histogram

    _edges, w = pitch_histogram(c, octave_wrap=True, quantize=True)
    p = w / max(w.sum(), 1e-9)
    p = p[p > 0]
    entropy = float(-(p * np.log2(p)).sum())
    folded = c % 1200.0
    near_sa = float(np.mean(np.minimum(folded, 1200.0 - folded) < 25.0))
    return {"voiced": frac, "conf": float(conf.mean()), "jitter": jitter,
            "entropy": entropy, "near_sa": near_sa}


def run(path, backends, out_dir=None, plot=False, sr_target=22050):
    import librosa

    audio, sr = librosa.load(path, sr=sr_target, mono=True)
    dur = len(audio) / sr
    print(f"\n=== {Path(path).name}  ({dur:.1f}s @ {sr} Hz)")
    hdr = (f"{'backend':<14} {'voiced%':>8} {'conf':>6} {'jitter¢':>8} {'%Sa':>6} "
           f"{'H(pitch)':>9} {'melody energy':>14}")
    print(hdr); print("-" * len(hdr))

    results = {}
    for b in backends:
        try:
            stems = separate(audio, sr, backend=b)
        except Exception as e:
            print(f"{b:<14} FAILED: {type(e).__name__}: {e}")
            continue
        m = pitch_metrics(stems.melody, sr)
        split = stems.energy_split()
        results[b] = (stems, m, split)
        print(f"{b:<14} {100*m['voiced']:8.1f} {m['conf']:6.3f} {m['jitter']:8.1f} "
              f"{100*m['near_sa']:6.1f} {m['entropy']:9.3f} {split['melody']:14.3f}")

        if out_dir:
            import soundfile as sf

            d = Path(out_dir) / Path(path).stem / b.replace(":", "_").replace("+", "_")
            d.mkdir(parents=True, exist_ok=True)
            for nm in ("melody", "percussion", "drone", "residual"):
                v = getattr(stems, nm)
                if v is not None and len(v):
                    sf.write(d / f"{nm}.wav", v, sr)
            sf.write(d.parent / "mixture.wav", audio, sr)

    if plot and results:
        _plot(path, audio, sr, results, out_dir)
    return results


def _plot(path, audio, sr, results, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import librosa
    import librosa.display

    n = len(results)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 7), squeeze=False)
    for j, (b, (stems, m, _split)) in enumerate(results.items()):
        S = librosa.amplitude_to_db(np.abs(librosa.stft(stems.melody, n_fft=2048)), ref=np.max)
        librosa.display.specshow(S, sr=sr, y_axis="log", x_axis="time", ax=axes[0][j])
        axes[0][j].set(title=f"{b} — melody stem", ylim=(60, 2000))

        f0, conf, hop, thr = track_pitch(stems.melody, sr)
        v = conf >= thr
        t = np.arange(len(f0)) * hop
        axes[1][j].plot(t[v], f0[v], ".", ms=1.5)
        axes[1][j].set(title=f"f0  ·  voiced {100*m['voiced']:.0f}%  H {m['entropy']:.2f}",
                       yscale="log", ylim=(60, 1000), xlabel="s", ylabel="Hz")
    fig.suptitle(Path(path).name)
    fig.tight_layout()
    dest = Path(out_dir or ".") / f"{Path(path).stem}_separation.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=110)
    plt.close(fig)
    print(f"  plot -> {dest}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", nargs="*", default=None, help="audio file(s)")
    ap.add_argument("--clip", default=None, help="raag folder name, sampled from the dataset")
    ap.add_argument("--n", type=int, default=2, help="how many clips to sample")
    ap.add_argument("--backends", nargs="+", default=None)
    ap.add_argument("--out", default=None, help="write stems (and plots) here")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--list", action="store_true", help="show which backends can run")
    args = ap.parse_args()

    if args.list:
        for k, ok in available().items():
            print(f"  {k:<14} {'available' if ok else 'NOT INSTALLED (pip install demucs)'}")
        return

    backends = args.backends or [b for b, ok in available().items() if ok]
    paths = list(args.audio or [])
    if args.clip or not paths:
        data = HERE.parent / "hindustani-raag-small-v1"
        pool = sorted((data / args.clip).glob("*.mp3")) if args.clip else \
            sorted(data.glob("*/*.mp3"))
        random.seed(0)
        paths += [str(p) for p in random.sample(pool, min(args.n, len(pool)))]
    for p in paths:
        run(p, backends, out_dir=args.out, plot=args.plot)


if __name__ == "__main__":
    main()
