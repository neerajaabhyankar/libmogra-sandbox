"""Fetch a pinned revision of a Hugging Face audio dataset to disk, and read it back.

An importable library first, a CLI second. Nothing here discovers a path by walking up
from `__file__`: you say which repo, which revision, and where the files go.

    from utils import dataset

    report = dataset.fetch(
        repo_id="neerajaabhyankar/hindustani-raag-small",
        revision="326caef0bc01da44ad46e4d9c65a5146da6bcc5b",
        audio_dir=Path("~/data/raag-v1.1").expanduser(),   # <Raag>/<clip>.mp3 written here
        tonics_csv=Path("~/data/raag-v1.1/tonics.csv").expanduser(),
    )
    clips = dataset.load_clips(audio_dir=report.audio_dir, tonics_csv=report.tonics_csv)

Every destination is a named parameter and every one is optional except `audio_dir` --
pass `tonics_csv=None` and no sidecar is written. `load_clips` reads only what you hand it.

Why this exists at all: for this dataset the v1 audio lives **only in the parquet files**.
The `<Raag>/*.mp3` tree at the repo root is still v0 and the `metadata.csv` the dataset
card mentions does not exist, so a loader that reads the raw layout silently gets v0 while
believing it has v1. Pinning a revision and materialising from parquet is what makes a
result reproducible.

CLI:

    python -m utils.dataset fetch  --version v1.1 --audio-dir ../hindustani-raag-small-v1.1
    python -m utils.dataset verify --version v1.1 --audio-dir ../hindustani-raag-small-v1.1
    python -m utils.dataset names  --version v1.1 --audio-dir ../hindustani-raag-small-v1.1
"""

import argparse
import csv
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import config

#: How this dataset names its clips. Used to *enrich* a record when it matches; a clip
#: whose name does not match is still fetched and still listed, with these fields None.
CLIP_RE = re.compile(r"^(train|test)_\[(.+)\]_chunk(\d+)\.mp3$")

TONICS_COLUMNS = ["clip_id", "raag", "split", "video", "chunk", "tonic_hz"]


@dataclass(frozen=True)
class Clip:
    """One audio file, plus whatever the dataset told us about it."""

    clip_id: str                 # "<Raag>/<filename>", stable across machines
    raag: str
    path: Path = None
    split: str = None            # train / test, if the filename says
    video: str = None            # source video id, if the filename says
    chunk: int = None
    tonic_hz: float = None       # None when the revision predates the tonic column


@dataclass
class FetchReport:
    repo_id: str
    revision: str
    audio_dir: Path
    tonics_csv: Path = None
    revision_file: Path = None
    n_clips: int = 0
    n_videos: int = 0
    n_unparsed: int = 0
    linked_from: Path = None
    unparsed: list = field(default_factory=list)

    def __str__(self):
        bits = [f"{self.n_clips} clips", f"{self.n_videos} videos", f"-> {self.audio_dir}"]
        if self.linked_from:
            bits.append(f"(audio symlinked from {self.linked_from})")
        if self.n_unparsed:
            bits.append(f"[{self.n_unparsed} filenames unparsed]")
        return " ".join(bits)


@dataclass
class VerifyReport:
    n_expected: int = 0
    missing: list = field(default_factory=list)
    differing: list = field(default_factory=list)

    @property
    def ok(self):
        return not self.missing and not self.differing

    def __str__(self):
        return (f"{self.n_expected} clips: {len(self.missing)} missing, "
                f"{len(self.differing)} differing")


# --------------------------------------------------------------- reading the Hub


def resolve_revision(repo_id, revision):
    """Confirm `revision` is a real, exact commit on `repo_id`. Returns the SHA.

    A branch or tag is refused rather than resolved: pinning to something that can move
    is how a past result silently changes meaning.
    """
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(repo_id, revision=revision)
    if info.sha != revision:
        raise ValueError(
            f"{revision!r} resolves to {info.sha} on {repo_id} -- pass the commit SHA, "
            f"not a branch or tag, so the pin cannot move under you."
        )
    return info.sha


def read_hub(repo_id, revision, split=None, audio_column="audio", label_column="label",
             tonic_column="tonic_hz"):
    """Yield `(raag, filename, audio_bytes, tonic_hz)` straight from the Hub.

    Uses `datasets.load_dataset`, so the parquet layout, sharding and label-name decoding
    are the library's problem rather than ours. Audio is read undecoded -- we want the
    original bytes to write back out, not a resampled float array.
    """
    from datasets import Audio, load_dataset

    ds = load_dataset(repo_id, revision=revision, split=split)
    if hasattr(ds, "keys"):                        # a DatasetDict -- chain the splits
        parts = [ds[k] for k in sorted(ds.keys())]
    else:
        parts = [ds]

    for part in parts:
        if audio_column not in part.column_names:
            raise KeyError(f"no {audio_column!r} column in {repo_id}@{revision[:10]}; "
                           f"columns are {part.column_names}")
        part = part.cast_column(audio_column, Audio(decode=False))
        names = None
        feat = part.features.get(label_column)
        if feat is not None and hasattr(feat, "names"):
            names = feat.names
        has_tonic = tonic_column in part.column_names

        for row in part:
            label = row.get(label_column)
            raag = names[int(label)] if (names and isinstance(label, int)) else str(label)
            audio = row[audio_column]
            yield (raag, Path(audio["path"]).name, audio["bytes"],
                   float(row[tonic_column]) if has_tonic and row[tonic_column] is not None
                   else None)


# --------------------------------------------------------------- writing to disk


def _parse_name(filename):
    m = CLIP_RE.match(filename)
    return (m.group(1), m.group(2), int(m.group(3))) if m else (None, None, None)


def fetch(repo_id, revision, audio_dir, tonics_csv=None, revision_file=None,
          link_audio_from=None, verify_link=True, progress=print):
    """Materialise `repo_id@revision` under `audio_dir`. Returns a `FetchReport`.

    audio_dir       where `<Raag>/<clip>.mp3` goes. Created if absent. Required.
    tonics_csv      where the per-clip sidecar goes. Skipped entirely if None.
    revision_file   where to stamp `<repo_id>@<sha>`. Skipped entirely if None.
    link_audio_from an already-fetched `audio_dir` whose bytes are identical; symlinked
                    per-raag instead of copied. Checked before use unless verify_link.
    """
    revision = resolve_revision(repo_id, revision)
    audio_dir = Path(audio_dir)
    rows = list(read_hub(repo_id, revision))
    if not rows:
        raise RuntimeError(f"{repo_id}@{revision[:10]} yielded no rows")
    progress(f"{repo_id}@{revision[:10]}: {len(rows)} clips")

    src = Path(link_audio_from) if link_audio_from else None
    if src and verify_link:
        bad = sum(1 for raag, fn, blob, _ in rows
                  if not (src / raag / fn).exists()
                  or hashlib.md5((src / raag / fn).read_bytes()).hexdigest()
                  != hashlib.md5(blob).hexdigest())
        if bad:
            progress(f"  {src} differs on {bad} clips -- copying instead of linking")
            src = None
        else:
            progress(f"  audio byte-identical to {src}; symlinking")

    report = FetchReport(repo_id=repo_id, revision=revision, audio_dir=audio_dir,
                         linked_from=src)
    audio_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for raag, filename, blob, tonic in rows:
        if src:
            link = audio_dir / raag
            if not link.exists():
                link.symlink_to((src / raag).resolve(), target_is_directory=True)
        else:
            dst = audio_dir / raag / filename
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(blob)
        split, video, chunk = _parse_name(filename)
        if split is None:
            report.unparsed.append(filename)
        written.append((f"{raag}/{filename}", raag, split, video, chunk, tonic))

    report.n_clips = len(written)
    report.n_videos = len({r[3] for r in written if r[3]})
    report.n_unparsed = len(report.unparsed)

    if tonics_csv:
        tonics_csv = Path(tonics_csv)
        tonics_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(tonics_csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(TONICS_COLUMNS)
            w.writerows(sorted(written))
        report.tonics_csv = tonics_csv
    if revision_file:
        revision_file = Path(revision_file)
        revision_file.parent.mkdir(parents=True, exist_ok=True)
        revision_file.write_text(f"{repo_id}@{revision}\n")
        report.revision_file = revision_file

    progress(str(report))
    if report.n_unparsed:
        progress(f"  unparsed filenames (fetched, but split/video/chunk are blank): "
                 f"{report.unparsed[:3]}{' ...' if report.n_unparsed > 3 else ''}")
    return report


def verify(repo_id, revision, audio_dir, progress=print):
    """Check what is on disk against the Hub, byte for byte. Writes nothing."""
    revision = resolve_revision(repo_id, revision)
    audio_dir = Path(audio_dir)
    report = VerifyReport()
    for raag, filename, blob, _ in read_hub(repo_id, revision):
        report.n_expected += 1
        f = audio_dir / raag / filename
        if not f.exists():
            report.missing.append(f"{raag}/{filename}")
        elif hashlib.md5(f.read_bytes()).hexdigest() != hashlib.md5(blob).hexdigest():
            report.differing.append(f"{raag}/{filename}")
    progress(str(report))
    return report


# --------------------------------------------------------------- reading it back


def load_clips(audio_dir=None, tonics_csv=None):
    """The dataset as `[Clip, ...]`, sorted by clip_id.

    Reads `tonics_csv` when given -- that is the authoritative list, written by `fetch`.
    Falls back to walking `audio_dir` only when no csv is supplied, and says so by leaving
    `tonic_hz` None. Give it at least one of the two.
    """
    if tonics_csv is None and audio_dir is None:
        raise ValueError("give load_clips a tonics_csv, an audio_dir, or both")
    audio_dir = Path(audio_dir) if audio_dir else None

    if tonics_csv:
        with open(tonics_csv) as fh:
            rows = list(csv.DictReader(fh))
        missing = set(TONICS_COLUMNS) - set(rows[0] if rows else [])
        if missing:
            raise ValueError(f"{tonics_csv} is missing columns {sorted(missing)}")
        return [
            Clip(clip_id=r["clip_id"], raag=r["raag"],
                 path=(audio_dir / r["clip_id"]) if audio_dir else None,
                 split=r["split"] or None, video=r["video"] or None,
                 chunk=int(r["chunk"]) if r["chunk"] else None,
                 tonic_hz=float(r["tonic_hz"]) if r["tonic_hz"] else None)
            for r in sorted(rows, key=lambda r: r["clip_id"])
        ]

    clips = []
    for raag_dir in sorted(p for p in audio_dir.iterdir() if p.is_dir()):
        for f in sorted(raag_dir.iterdir()):
            if f.suffix.lower() not in (".mp3", ".wav", ".flac", ".ogg", ".m4a"):
                continue
            split, video, chunk = _parse_name(f.name)
            clips.append(Clip(clip_id=f"{raag_dir.name}/{f.name}", raag=raag_dir.name,
                              path=f, split=split, video=video, chunk=chunk))
    return clips


def raag_names(tonics_csv=None, audio_dir=None):
    """The sorted class list. From the csv's `raag` column, or the audio dir's subfolders.

    This is the honest source for "which raags are in play". It replaces reading the
    class names off a *different* dataset version's directory tree, which is how a
    240 MB folder of audio ended up being a dependency of a metrics function.
    """
    return sorted({c.raag for c in load_clips(audio_dir=audio_dir, tonics_csv=tonics_csv)})


def read_revision_stamp(revision_file):
    """`(repo_id, sha)` from a stamp written by `fetch`, or None if absent/unreadable."""
    p = Path(revision_file)
    if not p.exists():
        return None
    text = p.read_text().strip()
    return tuple(text.split("@", 1)) if "@" in text else None


# --------------------------------------------------------------- CLI


def _add_target_args(p):
    p.add_argument("--repo-id", default=config.DATASET_ID)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--revision", help="exact commit SHA")
    g.add_argument("--version", choices=sorted(config.DATA_REVISIONS),
                   help=f"a pinned label from utils.config (default {config.DATA_VERSION})")
    p.add_argument("--audio-dir", required=True, type=Path)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m utils.dataset", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="materialise a revision to disk")
    _add_target_args(f)
    f.add_argument("--tonics-csv", type=Path, default=None,
                   help="default: <audio-dir>/tonics.csv; pass 'none' to skip")
    f.add_argument("--revision-file", type=Path, default=None,
                   help="default: <audio-dir>/REVISION; pass 'none' to skip")
    f.add_argument("--link-audio-from", type=Path, default=None,
                   help="reuse identical audio from an already-fetched directory")

    v = sub.add_parser("verify", help="check disk against the Hub, write nothing")
    _add_target_args(v)

    n = sub.add_parser("names", help="print the class list")
    _add_target_args(n)

    a = ap.parse_args(argv)
    rev = a.revision or config.revision(a.version)

    if a.cmd == "fetch":
        def _opt(value, default):
            if value is None:
                return default
            return None if str(value).lower() == "none" else value
        report = fetch(repo_id=a.repo_id, revision=rev, audio_dir=a.audio_dir,
                       tonics_csv=_opt(a.tonics_csv, a.audio_dir / "tonics.csv"),
                       revision_file=_opt(a.revision_file, a.audio_dir / "REVISION"),
                       link_audio_from=a.link_audio_from)
        return 0 if report.n_clips else 1
    if a.cmd == "verify":
        return 0 if verify(repo_id=a.repo_id, revision=rev, audio_dir=a.audio_dir).ok else 1
    if a.cmd == "names":
        csv_path = a.audio_dir / "tonics.csv"
        for name in raag_names(tonics_csv=csv_path if csv_path.exists() else None,
                               audio_dir=a.audio_dir):
            print(name)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
