"""Every results table in this project, from one loader, one deriver, one renderer.

Runs land in `results/v1.1/<run_id>/result.json` in a fixed schema, so tables are generated
and cannot drift from what was measured. A new table is a row in `TABLES` -- never another
JSON walk and another f-string.

    table("full")                      -> results/v1.1/RESULTS.md
    table("notebook", order=[...])     -> a plan.md entry, same shape every time
    table("status")                    -> the ranked list scripts/status.sh prints
"""

import json
import math

from .paths import RESULTS

#: Each architecture's Stage 1 run: the baseline "vs stage 1" always means.
BASELINE = {"cqt": "c1", "resnet1d": "r1", "hubert": "d1"}

#: Short descriptions, so a run reads the same wherever it appears.
WHAT = {
    "c1": "CQT, fixed fmin", "c2": "CQT, Sa-anchored",
    "c2_shuffled": "c2, tonics permuted *(control)*", "c3": "c2 + HPSS melody stem",
    "c4g": "c2 + graded label smoothing", "c4a": "c2 + auxiliary occupancy head",
    "c4h": "c2 + **DB-template head**",
    "r1": "jeevster ResNet, as-is", "r2n": "ResNet, tonic-normalised audio",
    "r2c": "ResNet, tonic by FiLM", "r2c_shuffled": "r2c, tonic permuted *(control)*",
    "r3": "r2n + HPSS melody stem", "r4g": "r2n + graded label smoothing",
    "d1": "distilHuBERT, notebook recipe", "d2n": "distilHuBERT, tonic-normalised audio",
    "d2c": "distilHuBERT, tonic by FiLM", "d1_unfrozen": "d1, conv encoder unfrozen",
    # Batch 4 onward: descriptive ids instead of codes -- c4h etc. were getting unreadable
    "dbprior_lam0": "c4h, learned templates (`--db-lam 0`) *(ablation)*",
    "dbprior_36bins": "c4h at 36 swar bins (~33 cents)",
    "dbprior_144bins": "c4h at 144 swar bins",
    "dbprior_frozen": "c4h, database templates frozen",
    "aug_jitter": "c4h + pitch/gain jitter",
    "seed1": "c4h at seed 1", "seed2": "c4h at seed 2",
    "cv5": "c4h, 5-fold grouped CV",
    "aug_seed1": "aug_jitter at seed 1", "aug_seed2": "aug_jitter at seed 2",
    "fuse_aug_jitter_m14": "aug_jitter + M14, probability fusion",
    "fuse_c4h_m14": "c4h + M14, probability fusion",
    "fuse_aug_seed1_m14": "aug_jitter seed 1 + M14", "fuse_aug_seed2_m14": "aug_jitter seed 2 + M14",
    "melody_only": "melody histogram alone, logreg *(control)*",
    "melody_only_seed1": "melody histogram alone, seed 1",
    "melody_only_seed2": "melody histogram alone, seed 2",
    "hybrid_feat": "aug_jitter + **melody histogram as an input**",
    "hybrid_seed1": "hybrid_feat at seed 1", "hybrid_seed2": "hybrid_feat at seed 2",
    "hybrid_nodb": "hybrid_feat without the DB-template head",
}

#: name -> (columns as (field, heading, format), markdown?)
TABLES = {
    "full": ([("run", "run", "{}"), ("arch", "arch", "{}"), ("tonic", "tonic", "{}"),
              ("sep", "sep", "{}"), ("prior", "DB prior", "{}"), ("split", "split", "{}"),
              ("val", "top-1", "{:.3f}"), ("top5", "top-5", "{:.3f}"),
              ("mrr", "MRR", "{:.3f}"), ("f1", "macro-F1", "{:.3f}"),
              ("vote", "video", "{:.3f}"), ("test", "test top-1", "{}"),
              ("mins", "min", "{:.0f}")], True),
    "notebook": ([("run", "run", "{}"), ("what", "what", "{}"),
                  ("val", "val top-1", "{:.3f}"), ("test", "test top-1", "{}"),
                  ("delta", "vs stage 1", "{}"),
                  ("aff", "mistake affinity (chance)", "{}")], True),
    "status": ([("val", "val", "{:.3f}"), ("test", "test", "{}"), ("run", "run", "{}"),
                ("arch", "arch", "{}"), ("stage", "stage", "{}")], False),
}


def load_runs(results_dir=None):
    """Every readable `result.json` under the results directory."""
    runs = []
    for f in sorted((results_dir or RESULTS).glob("*/result.json")):
        try:
            runs.append(json.loads(f.read_text()))
        except json.JSONDecodeError:
            print(f"  skipping unreadable {f}")
    return runs


def derive(r, by_id=None):
    """One `result.json` -> every field any table needs. The only place that knows the
    schema; add a field here rather than reaching into the json somewhere else."""
    cfg, m, mu = r.get("config", {}), r.get("metrics", {}), r.get("musical", {})
    tag = lambda cond, s: [s] if cond else []                            # noqa: E731
    tonic = (tag(cfg.get("tonic") == "normalise", "audio")
             + tag(cfg.get("tonic_mode") == "condition", "FiLM")
             + tag(cfg.get("shuffle_tonics"), "SHUFFLED"))
    prior = (tag(cfg.get("graded_alpha"), f"graded {cfg.get('graded_alpha')}")
             + tag(cfg.get("aux_weight"), f"aux {cfg.get('aux_weight')}")
             + tag(cfg.get("db_head"), f"DB head lam={cfg.get('db_lam')}"))

    val = m.get("top1", float("nan"))
    test = r.get("test", {}).get("metrics", {}).get("top1")
    base = (by_id or {}).get(BASELINE.get(r.get("arch")), {}).get("metrics", {}).get("top1")
    aff, chance = mu.get("mistake_affinity"), mu.get("mistake_affinity_chance")

    return {
        "run": r.get("run_id", "?"), "what": WHAT.get(r.get("run_id"), r.get("run_id", "?")),
        "arch": r.get("arch", "?"), "stage": r.get("stage", "?"),
        "tonic": "+".join(tonic) or "-", "sep": cfg.get("separate") or "-",
        "prior": ", ".join(prior) or "-", "split": r.get("split", "?").replace("grouped-", ""),
        "val": val, "top5": m.get("top5", float("nan")), "mrr": m.get("mrr", float("nan")),
        "f1": m.get("macro_f1", float("nan")), "vote": m.get("video_vote", float("nan")),
        "test": f"{test:.3f}" if test is not None else "-",
        "gap": f"{val - test:+.3f}" if test is not None and not math.isnan(val) else "-",
        "delta": ("—" if BASELINE.get(r.get("arch")) == r.get("run_id")
                  else f"{val - base:+.3f}" if base is not None and not math.isnan(val)
                  else "-"),
        "aff": f"{aff:.3f} ({chance:.3f})" if aff is not None else "-",
        "mins": r.get("wall_clock_s", 0) / 60.0,
    }


def table(name, runs=None, order=None, sort_by="val"):
    """A named table. `order` fixes the rows by run id; otherwise sorted by `sort_by`."""
    columns, markdown = TABLES[name]
    runs = load_runs() if runs is None else runs
    by_id = {r.get("run_id"): r for r in runs}

    if order is None:
        rows = sorted((derive(r, by_id) for r in runs),
                      key=lambda r: r["val"] if isinstance(r["val"], float) else 0,
                      reverse=True)
    else:                       # a missing run is shown as missing, never silently dropped
        rows = [derive(by_id[i], by_id) if i in by_id
                else {**{k: "-" for k, _h, _f in columns}, "run": i,
                      "what": WHAT.get(i, i), "val": float("nan"), "delta": "*not run*"}
                for i in order]

    def cell(row, key, fmt):
        try:
            return fmt.format(row.get(key, "-"))
        except (ValueError, TypeError):
            return str(row.get(key, "-"))

    body = [[cell(r, k, f) for k, _h, f in columns] for r in rows]
    if not markdown:
        return "\n".join("  ".join(c) for c in body)
    head = [h for _k, h, _f in columns]
    return "\n".join(["| " + " | ".join(head) + " |",
                      "|" + "|".join("---" for _ in columns) + "|"]
                     + ["| " + " | ".join(c) + " |" for c in body])


def write_results_md(runs=None, path=None):
    path = path or (RESULTS / "RESULTS.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([
        "# Results — dataset v1.1", "",
        "Generated by `scripts/90_report.py`. Do not hand-edit; re-run the script.", "",
        "Chance = 0.020. `split` is how the non-test column was measured: `val` is one "
        "video-grouped 20 % split of the 1810 train clips, `Nfold-cv` is pooled "
        "out-of-fold predictions over all of them. `test top-1` is the held-out 150 clips, "
        "video-disjoint from everything above.", "",
        table("full", runs), "",
    ]))
    return path
