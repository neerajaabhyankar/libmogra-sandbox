"""Regenerate results/v1.1/RESULTS.md from every run's result.json."""

import argparse

import _bootstrap  # noqa: F401
from common.report import load_runs, markdown_table, write_results_md

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--write", action="store_true", help="also refresh RESULTS.md")
args = ap.parse_args()

runs = load_runs()
print(f"{len(runs)} runs\n")
print(markdown_table(runs))
if args.write:
    print(f"\nwrote {write_results_md(runs)}")
