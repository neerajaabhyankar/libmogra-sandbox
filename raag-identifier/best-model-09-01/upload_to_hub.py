"""Push this directory to the Hugging Face Hub as a model repo.

    python upload_to_hub.py --repo-id <you>/cqt-histogram-hindustani-raag-small --dry-run
    python upload_to_hub.py --repo-id <you>/cqt-histogram-hindustani-raag-small

Nothing is uploaded without `--repo-id`, and `--dry-run` lists exactly what would go. Log
in first with `huggingface-cli login`, or pass `--token`.

What gets uploaded: the code, the weights, the model card and its figure. What does not:
`cache/` (250 MB of derived features, rebuildable by `train.py`), `__pycache__`, and any
audio you happen to have left lying here.
"""

import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent

#: Everything under here is derived, private, or huge. Kept out of the repo.
#: fnmatch's `*` crosses directory separators, so a bare `*.pyc` catches nested ones too --
#: but a pattern with no wildcard at all (`.DS_Store`) only ever matches at the root.
IGNORE = ["cache/*", "*__pycache__*", "*.pyc", "*.DS_Store",
          "*.wav", "*.mp3", "*.flac"]


def files_that_would_upload(root=HERE):
    import fnmatch

    out = []
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root).as_posix()
        if p.is_dir() or any(fnmatch.fnmatch(rel, pat) for pat in IGNORE):
            continue
        out.append((rel, p.stat().st_size))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", help="e.g. neerajaabhyankar/cqt-histogram-hindustani-raag-small")
    ap.add_argument("--token", default=None, help="defaults to your cached login")
    ap.add_argument("--private", action="store_true", help="create the repo private")
    ap.add_argument("--message", default="Upload raag fusion model")
    ap.add_argument("--dry-run", action="store_true", help="list the files and stop")
    a = ap.parse_args()

    files = files_that_would_upload()
    total = sum(size for _rel, size in files)
    for rel, size in files:
        print(f"  {size / 1e6:8.2f} MB  {rel}")
    print(f"  {'-' * 8}\n  {total / 1e6:8.2f} MB  in {len(files)} files")

    if a.dry_run or not a.repo_id:
        print("\nnothing uploaded." + ("" if a.repo_id else " Pass --repo-id to upload."))
        return

    from huggingface_hub import HfApi

    api = HfApi(token=a.token)
    api.create_repo(a.repo_id, repo_type="model", private=a.private, exist_ok=True)
    api.upload_folder(repo_id=a.repo_id, repo_type="model", folder_path=str(HERE),
                      ignore_patterns=IGNORE, commit_message=a.message)
    print(f"\nhttps://huggingface.co/{a.repo_id}")


if __name__ == "__main__":
    main()
