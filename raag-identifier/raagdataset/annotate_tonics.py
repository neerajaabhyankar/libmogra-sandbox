"""Hand-annotate the true tonic (Sa) of every video in `hindustani-raag-small`.

Why this exists: in `motif-classifier`, replacing the estimated tonic with an oracle one
takes 50-way top-1 from 0.109 to 0.314 and top-5 from 0.277 to 0.721. Sa placement is worth
roughly 3x every other modelling choice combined, and no automatic estimator tried so far
gets close. So the tonic becomes a labelled field of the dataset.

## What one round looks like

For each video (not each clip — chunks of a video share a recording and therefore a tonic):

  1. a 10 s excerpt from the middle of one of its chunks plays;
  2. the YouTube URL is printed, in case you want more context than 10 s gives;
  3. you hum Sa into the mic for a few seconds;
  4. your hum is pitch-tracked, then **snapped to the nearest strong peak in the video's own
     pitch histogram**, so being 10-20 cents flat does not become 10-20 cents of label noise;
  5. the result is written to `tonics.csv` immediately.

Snapping is the point of step 4. What we want labelled is the recording's Sa, and the
recording states it far more precisely than a hummed approximation — the hum only has to be
accurate enough to pick the *right* peak (and the right octave). If no peak lies within
`--snap-cents` the raw hum is kept and the row is flagged, so it can be reviewed later.

## Resumability

`tonics.csv` is appended after every single annotation and re-read on startup; already-done
videos are skipped. Stop with Ctrl-C or `q` at any prompt and rerun later — nothing is lost
and nothing is redone. `--redo` revisits videos already annotated.

## Usage

    poetry run python annotate.py                  # annotate everything not yet done
    poetry run python annotate.py --raag Yaman     # just one raag
    poetry run python annotate.py --limit 20       # a short session
    poetry run python annotate.py --status         # how far along am I
    poetry run python annotate.py --redo --video ABC123   # fix one entry
    poetry run python annotate.py --review         # replay each annotation to check it

Nothing here uploads or pushes anything; `tonics.csv` is a local file.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "hindustani-raag-small"
MELODY_DIR = HERE.parent / "melody-extraction"
OUT_CSV = HERE / "tonics.csv"

for p in (str(MELODY_DIR),):
    if p not in sys.path:
        sys.path.insert(0, p)

SAMPLE_RATE = 22050
HUM_SECONDS = 4.0
EXCERPT_SECONDS = 10.0
FIELDS = ["video", "raag", "tonic_hz", "note", "hum_hz", "snap_cents", "snapped", "clip", "timestamp"]


# ---------------------------------------------------------------- dataset


def videos():
    """{video_id: {"raag": str, "clips": [Path, ...]}} — chunks grouped by source video."""
    import re

    pat = re.compile(r"^(train|test)_\[(.+)\]_chunk(\d+)\.mp3$")
    out = {}
    for raag_dir in sorted(DATA_DIR.iterdir()):
        if not raag_dir.is_dir():
            continue
        for f in sorted(raag_dir.iterdir()):
            m = pat.match(f.name)
            if not m:
                continue
            entry = out.setdefault(m.group(2), {"raag": raag_dir.name, "clips": []})
            entry["clips"].append(f)
    return out


def url_for(video_id):
    return f"https://www.youtube.com/watch?v={video_id}"


# ---------------------------------------------------------------- storage


def load_done():
    if not OUT_CSV.exists():
        return {}
    with OUT_CSV.open() as fh:
        return {r["video"]: r for r in csv.DictReader(fh) if r.get("video")}


def append_row(row):
    """Append-and-flush after every annotation, so a Ctrl-C never costs more than nothing."""
    new = not OUT_CSV.exists()
    with OUT_CSV.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
        fh.flush()


def rewrite(rows):
    with OUT_CSV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows.values():
            w.writerow(r)


def load_skip_list():
    return [
        "0GkzyJbxCMA",
        "5z_sUCM__uc",
        "ZWQDBjkAW-w",
        "PKUO-nqbABg",
        "A3SneqtmNog",
        "azznhT3coJE",
        "FZtu7xweL0Y",
        "M3DE88z0Nv8",
        "MN7VkVPaytM",
        "85y7SzjvCjk",
        "aUWTwQk_kUY",
        "e3iMnt28DLM",
        "k86EmhqcpUA",
        "nI7v5zCLawk",
        "2gJbTYxmqMA",
        "LknyChMkj3g",
        "iOtSjWMKdZI",
        "AVxwGAy1ygM",
        "G_wmgnxgK0w",
        "egHCxISQG9o",
    ]


# ---------------------------------------------------------------- audio


def play(audio, sr, label=""):
    import sounddevice as sd

    if label:
        print(f"  ♪ {label}")
    sd.play(audio, sr)
    sd.wait()


def excerpt(clip_path, seconds=EXCERPT_SECONDS):
    """`seconds` from the middle of the clip — the start is often applause or an announcement."""
    import librosa

    audio, sr = librosa.load(str(clip_path), sr=None, mono=True)
    n = int(seconds * sr)
    if len(audio) > n:
        start = (len(audio) - n) // 2
        audio = audio[start : start + n]
    return audio, sr


def record(duration=HUM_SECONDS, sr=SAMPLE_RATE):
    """Record from the mic with a progress bar (same pattern as melody-extraction/main_live)."""
    import sounddevice as sd

    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
    width = 40
    start = time.time()
    while (elapsed := time.time() - start) < duration:
        filled = int(width * min(elapsed / duration, 1.0))
        sys.stdout.write(f"\r  [{'█' * filled}{'░' * (width - filled)}] {elapsed:.1f}s / {duration:.1f}s")
        sys.stdout.flush()
        time.sleep(0.05)
    sys.stdout.write(f"\r  [{'█' * width}] {duration:.1f}s / {duration:.1f}s\n")
    sd.wait()
    return audio.reshape(-1), sr


def hum_to_hz(audio, sr):
    """Pitch of the hum: median of pyin's voiced frames.

    Median rather than melody-extraction's mode-of-distribution heuristic — that one snaps to
    a semitone of the A440 grid, which would throw away up to 50 cents before we have even
    started, and a sustained hum has no distribution to take a mode of anyway.
    """
    import librosa

    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    f0, voiced, _ = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=SAMPLE_RATE,
        frame_length=2048,
        hop_length=512,
        fill_na=0.0,
    )
    ok = voiced & (f0 > 0)
    if ok.sum() < 3:
        return None
    return float(np.median(f0[ok]))


# ---------------------------------------------------------------- snapping


def pitch_histogram(clips, bin_cents=10.0, ref_hz=55.0, max_clips=3):
    """Fine-grained pitch histogram over the video's own audio, in cents above `ref_hz`.

    10-cent bins, deliberately much finer than a semitone: the whole point is to land on the
    recording's actual Sa, which is wherever the tanpura is tuned, not on an A440 grid.
    """
    import librosa

    from trackers.tony.tony_tracker import _run_pyin, TONY_PARAMETERS

    edges, weights = [], []
    for clip in clips[:max_clips]:
        audio, sr = librosa.load(str(clip), sr=None, mono=True)
        f0, voiced, _hop, _notes = _run_pyin(audio, sr, dict(TONY_PARAMETERS))
        f = f0[voiced & (f0 > 0)]
        if len(f):
            edges.append(1200.0 * np.log2(f / ref_hz))
            weights.append(np.ones(len(f)))
    if not edges:
        return None, None
    cents = np.concatenate(edges)
    lo, hi = 0.0, 1200.0 * np.log2(2000.0 / ref_hz)
    n = int((hi - lo) / bin_cents)
    hist, bin_edges = np.histogram(cents, bins=n, range=(lo, hi))
    centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    # light smoothing so vibrato does not split one note into two peaks
    k = np.exp(-0.5 * (np.arange(-6, 7) / 2.0) ** 2)
    hist = np.convolve(hist.astype(float), k / k.sum(), mode="same")
    return centres, hist


TONIC_BAND = (95.0, 260.0)  # where a sung Sa actually lives; used only to fix the octave


def canonical_octave(hz, band=TONIC_BAND, centres=None, hist=None, ref_hz=55.0):
    """Fold a tonic into the standard octave band, resolving ties by histogram support.

    Sa hummed an octave up is still Sa, but the *label* has to be octave-consistent across
    the dataset or downstream code silently gets a 2x error. Folding is exact arithmetic —
    halving stays on the same pitch class — so this can never move the label off Sa.

    The band (95-260 Hz) deliberately spans more than one octave, because a male and a
    female Sa genuinely differ by that much, so folding alone can leave *two* candidates in
    range. When that happens the recording decides: whichever octave carries more mass in
    the pitch histogram is the one the performance actually treats as Sa.
    """
    lo, hi = band
    if hz <= 0:
        return hz
    while hz < lo:
        hz *= 2.0
    while hz >= hi:
        hz /= 2.0

    candidates = [hz]
    f = hz * 2.0
    while f < hi:
        candidates.append(f)
        f *= 2.0
    f = hz / 2.0
    while f >= lo:
        candidates.append(f)
        f /= 2.0
    if len(candidates) == 1 or centres is None or hist is None:
        return hz

    def support(f):
        c = 1200.0 * np.log2(f / ref_hz)
        i = int(np.argmin(np.abs(centres - c)))
        return float(hist[max(i - 1, 0) : i + 2].sum())

    return max(candidates, key=support)


def snap(hum_hz, centres, hist, ref_hz=55.0, max_cents=60.0, min_prominence=0.15):
    """Move the hum to the nearest prominent peak of the recording's pitch histogram.

    Returns (tonic_hz, offset_cents, snapped_bool). Peaks below `min_prominence` of the
    strongest peak are ignored so the hum cannot latch onto histogram noise. Octave errors in
    the hum are tolerated: candidates are considered at +/- 1 and 2 octaves too, and the
    label is written in the octave the recording actually supports.
    """
    if centres is None or hist is None or hist.max() <= 0:
        return hum_hz, 0.0, False
    peaks = [
        i for i in range(1, len(hist) - 1)
        if hist[i] >= hist[i - 1] and hist[i] > hist[i + 1] and hist[i] >= min_prominence * hist.max()
    ]
    if not peaks:
        return hum_hz, 0.0, False
    hum_cents = 1200.0 * np.log2(hum_hz / ref_hz)

    def refine(i):
        """Parabolic interpolation through the peak and its neighbours.

        Without this the label is pinned to the 10-cent bin grid, which throws away most of
        the precision snapping exists to capture."""
        y0, y1, y2 = hist[i - 1], hist[i], hist[i + 1]
        denom = y0 - 2 * y1 + y2
        delta = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
        delta = float(np.clip(delta, -1.0, 1.0))
        step = centres[1] - centres[0]
        return centres[i] + delta * step

    def nearest(target):
        best = None
        for i in peaks:
            c = refine(i)
            d = c - target
            if abs(d) <= max_cents and (best is None or abs(d) < abs(best[1])):
                best = (c, d)
        return best

    # Octave shifts are a *rescue*, not a competitor: try the octave actually hummed first
    # and only look elsewhere if nothing there is close enough. Searching all octaves at
    # once lets a peak an octave away win on a marginally smaller offset, which silently
    # relabels a correct hum into the wrong octave.
    for octave in (0, -1200, 1200, -2400, 2400):
        best = nearest(hum_cents + octave)
        if best is not None:
            snapped_hz = float(ref_hz * 2.0 ** (best[0] / 1200.0))
            return canonical_octave(snapped_hz, centres=centres, hist=hist, ref_hz=ref_hz), float(best[1]), True
    return canonical_octave(hum_hz, centres=centres, hist=hist, ref_hz=ref_hz), 0.0, False


# ---------------------------------------------------------------- driver


def annotate_one(video_id, info, args):
    """Returns a row dict, or None if skipped / quit."""
    import librosa

    clip = info["clips"][len(info["clips"]) // 2]
    print(f"\n{'=' * 72}")
    print(f"  raag   {info['raag']}")
    print(f"  video  {video_id}   {url_for(video_id)}")
    print(f"  clip   {clip.name}   ({len(info['clips'])} chunks share this tonic)")

    audio, sr = excerpt(clip)
    centres = hist = None

    while True:
        play(audio, sr, "playing 10 s from the middle of the clip")
        ans = input("  [Enter] hum Sa · [r] replay · [l] longer excerpt · [s] skip · [q] quit  > ").strip().lower()
        if ans == "r":
            continue
        if ans == "l":
            audio, sr = excerpt(clip, EXCERPT_SECONDS * 3)
            continue
        if ans == "s":
            print("  skipped")
            return None
        if ans == "q":
            raise KeyboardInterrupt
        break

    print(f"\n  hum and hold Sa for {HUM_SECONDS:.0f}s...")
    hum_audio, hum_sr = record()
    hum_hz = hum_to_hz(hum_audio, hum_sr)
    if hum_hz is None:
        print("  could not hear a pitch — try again")
        return annotate_one(video_id, info, args)

    if centres is None:
        print("  matching against the recording's pitch histogram...")
        centres, hist = pitch_histogram(info["clips"])
    tonic_hz, offset, snapped = snap(hum_hz, centres, hist, max_cents=args.snap_cents)

    note = librosa.hz_to_note(tonic_hz)
    print(f"\n  hummed  {hum_hz:7.2f} Hz ({librosa.hz_to_note(hum_hz)})")
    if snapped:
        print(f"  snapped {tonic_hz:7.2f} Hz ({note})   moved {offset:+.1f} cents onto a peak in the recording")
    else:
        print(f"  NOT snapped — no histogram peak within {args.snap_cents:.0f} cents; keeping the raw hum")

    while True:
        ans = input("  [Enter] accept · [p] play Sa against the clip · [a] again · [s] skip · [q] quit  > ").strip().lower()
        if ans == "p":
            t = np.arange(int(3 * sr)) / sr
            tone = 0.25 * np.sin(2 * np.pi * tonic_hz * t)
            mix = audio[: len(tone)] * 0.7
            play(mix + tone[: len(mix)], sr, "clip + a sine at the chosen Sa — they should agree")
            continue
        if ans == "a":
            return annotate_one(video_id, info, args)
        if ans == "s":
            return None
        if ans == "q":
            raise KeyboardInterrupt
        break

    return {
        "video": video_id,
        "raag": info["raag"],
        "tonic_hz": f"{tonic_hz:.4f}",
        "note": note,
        "hum_hz": f"{hum_hz:.4f}",
        "snap_cents": f"{offset:.1f}",
        "snapped": "yes" if snapped else "NO",
        "clip": clip.name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def status(vids, done):
    by_raag = {}
    for v, info in vids.items():
        d = by_raag.setdefault(info["raag"], [0, 0])
        d[1] += 1
        d[0] += v in done
    total_done = sum(d[0] for d in by_raag.values())
    print(f"{total_done} / {len(vids)} videos annotated across {len(by_raag)} raags")
    unsnapped = [r for r in done.values() if r.get("snapped") == "NO"]
    if unsnapped:
        print(f"  {len(unsnapped)} kept the raw hum (no histogram peak matched) — worth reviewing:")
        for r in unsnapped[:10]:
            print(f"    {r['raag']:16s} {r['video']}  {r['tonic_hz']} Hz")
    incomplete = {r: d for r, d in sorted(by_raag.items()) if d[0] < d[1]}
    if incomplete:
        print("  remaining: " + ", ".join(f"{r} {d[0]}/{d[1]}" for r, d in incomplete.items()))
    else:
        print("  all done")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raag", help="only this raag folder")
    ap.add_argument("--video", help="only this video id")
    ap.add_argument("--limit", type=int, help="stop after this many annotations this session")
    ap.add_argument("--redo", action="store_true", help="revisit videos already annotated")
    ap.add_argument("--status", action="store_true", help="print progress and exit")
    ap.add_argument("--review", action="store_true", help="replay existing annotations to check them")
    ap.add_argument("--snap-cents", type=float, default=60.0,
                    help="how far the hum may be moved onto a histogram peak (default 60)")
    args = ap.parse_args()

    vids = videos()
    skip = load_skip_list()
    done = load_done()

    if args.status:
        status(vids, done)
        return

    if args.review:
        review(vids, done, args)
        return

    todo = [
        (v, i) for v, i in sorted(vids.items(), key=lambda kv: (kv[1]["raag"], kv[0]))
        if (args.redo or v not in done)
        and (not args.raag or i["raag"] == args.raag)
        and (not args.video or v == args.video)
    ]
    if not todo:
        print("nothing to do — everything matching is already annotated (use --redo to revisit)")
        status(vids, done)
        return

    print(f"{len(todo)} videos to annotate ({len(done)} already done). Ctrl-C or 'q' stops safely.")
    n = 0
    try:
        for video_id, info in todo:
            if video_id in skip:
                print(f"\n{'=' * 72}\n  skipping {video_id} (in skip list)")
                continue
            row = annotate_one(video_id, info, args)
            if row is None:
                continue
            if args.redo and video_id in done:
                done[video_id] = row
                rewrite(done)
            else:
                append_row(row)
                done[video_id] = row
            n += 1
            print(f"  saved -> {OUT_CSV.name}   ({len(done)}/{len(vids)} done)")
            if args.limit and n >= args.limit:
                print("\nreached --limit for this session")
                break
    except KeyboardInterrupt:
        print("\n\nstopped — everything annotated so far is saved")
    status(vids, done)


def review(vids, done, args):
    """Replay each annotated video with a sine at the stored Sa, to catch bad rows."""
    rows = [r for r in done.values()
            if (not args.raag or r["raag"] == args.raag) and (not args.video or r["video"] == args.video)]
    if not rows:
        print("nothing annotated yet")
        return
    print(f"reviewing {len(rows)} annotations — [Enter] next, [q] quit")
    try:
        for r in rows:
            info = vids.get(r["video"])
            if not info:
                continue
            clip = next((c for c in info["clips"] if c.name == r["clip"]), info["clips"][0])
            audio, sr = excerpt(clip)
            hz = float(r["tonic_hz"])
            t = np.arange(int(min(6.0, len(audio) / sr) * sr)) / sr
            tone = 0.25 * np.sin(2 * np.pi * hz * t)
            print(f"\n  {r['raag']:16s} {r['video']}  {hz:.2f} Hz ({r['note']})  snapped={r['snapped']}")
            play(audio[: len(tone)] * 0.7 + tone, sr)
            if input("  > ").strip().lower() == "q":
                break
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
