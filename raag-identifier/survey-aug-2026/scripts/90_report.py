"""Results tables, generated from every run's result.json -- never hand-typed.

    python scripts/90_report.py                              # the full table
    python scripts/90_report.py --write                      # also refresh RESULTS.md
    python scripts/90_report.py --notebook c1 c2 c2_shuffled # a plan.md entry's table
"""

import argparse

import _bootstrap  # noqa: F401
from common.report import load_runs, table, write_results_md

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--write", action="store_true", help="also refresh RESULTS.md")
ap.add_argument("--notebook", nargs="+", metavar="RUN_ID",
                help="canonical plan.md table for these runs, in this order")
args = ap.parse_args()

runs = load_runs()
if args.notebook:
    print(table("notebook", runs, order=args.notebook))
else:
    print(f"{len(runs)} runs\n")
    print(table("full", runs))
    if args.write:
        print(f"\nwrote {write_results_md(runs)}")
