"""Evaluate one ../transcriber model as pakad-matcher's pitch source, with the usual scripts.

    poetry run python -m transcribers.run --source crepe                 # every step below
    poetry run python -m transcribers.run --source crepe --steps reader phrase

  transcribe  the model over segments.ranges() (cached in ../transcriber/cache/<source>/)
  reader      fit_reader.py --save          reader refit on notation, misread held out by recording
  phrase      s7.py --val                   phrase methods chosen on validation
  insights    insights.evaluate --val       insight methods chosen on train + validation

Each step runs the unchanged script with PAKAD_PITCH_SOURCE=<source>, so its outputs land in
transcribers/<source>/results/ and logs in transcribers/<source>/logs/. Test is never scored
here; the final pick gets `--steps test` once (segments.py "test", then s7.py --test and
insights.evaluate --test).
"""

import argparse
import os
import subprocess
import sys

import config as C

STEPS = {
    "reader": [sys.executable, "fit_reader.py", "--save"],
    "phrase": [sys.executable, "s7.py", "--val"],
    "insights": [sys.executable, "-m", "insights.evaluate", "--val"],
}
TEST_STEPS = {
    "test-phrase": [sys.executable, "s7.py", "--test"],
    "test-insights": [sys.executable, "-m", "insights.evaluate", "--test"],
}


def transcribe(source, purposes):
    import fullaudio
    from transcribers import segments, source as src  # noqa: F401  (puts ../ on sys.path)
    from transcriber import cache
    rs, idx = segments.ranges(purposes or segments.PURPOSES), fullaudio.index()
    for i, (video, r) in enumerate(sorted(rs.items()), 1):
        new = cache.transcribe(source, video, idx[video].path, r)
        print(f"  {i}/{len(rs)} {video}: {len(r)} ranges, {new} new", flush=True)


def run(source, step, cmd):
    out = C.TRANSCRIBERS_DIR / source
    (out / "logs").mkdir(parents=True, exist_ok=True)
    (out / "results" / "insights").mkdir(parents=True, exist_ok=True)
    log = out / "logs" / f"{step}.log"
    print(f"{step}: {' '.join(cmd[1:])} -> {log}", flush=True)
    with open(log, "w") as fh:
        subprocess.run(cmd, cwd=C.HERE, check=True, stdout=fh, stderr=subprocess.STDOUT,
                       env={**os.environ, "PAKAD_PITCH_SOURCE": source})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--steps", nargs="+", default=["transcribe", *STEPS])
    a = ap.parse_args()
    for step in a.steps:
        if step == "transcribe":
            transcribe(a.source, None)
        elif step == "test":
            transcribe(a.source, ("test",))
            for s, cmd in TEST_STEPS.items():
                run(a.source, s, cmd)
        else:
            run(a.source, step, STEPS[step])


if __name__ == "__main__":
    main()
