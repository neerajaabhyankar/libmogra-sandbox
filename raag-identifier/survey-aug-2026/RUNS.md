# RUNS.md — the run registry

Bookkeeping only: **what is queued, what is running, what is done, and how to drive it.**
No analysis lives here — findings go in [`plan.md`](plan.md), numbers in
[`results/v1.1/RESULTS.md`](results/v1.1/RESULTS.md).

Status: 🟨 ready, not started · 🔄 in progress · ✅ done · 🟥 not ready

A row is 🔄 only while a process is actually alive. An interrupted batch goes back to 🟨:
partial epochs survive in `state.pt` and are reused, but a run has produced nothing until
`result.json` exists.

---

## How to drive it

```bash
cd /Users/neerajaabhyankar/Repos/icm-shruti-analysis/raag-identifier/survey-aug-2026

nohup bash scripts/run_batch1_cheap.sh   > /tmp/batch1.out 2>&1 &   # Batch 1
nohup bash scripts/run_batch2_hubert.sh  > /tmp/batch2.out 2>&1 &   # Batch 2 (or Colab)
nohup bash scripts/run_batch3_sep_db.sh  > /tmp/batch3.out 2>&1 &   # Batch 3
```

`nohup … &` matters: in the foreground, closing the terminal kills the job.

- A run with a `result.json` is **skipped**. To redo one, delete `results/v1.1/<id>/`
  first, or `FORCE=1 bash scripts/<batch>.sh` to redo all of them.
- Everything is **resumable** — re-running picks up from `state.pt` and rebuilds no cache
  it already has.
- Writes only under `results/v1.1/<run_id>/` and `cache/`. Never calls `push_to_hub`.

**What is running right now — one command, no run id needed:**

```bash
bash scripts/status.sh
```

Prints the live run(s) with the current epoch, best top-1 so far and the exact `tail -f`
for each; then everything that has reported, ranked; then anything part-finished that will
resume on the next launch. `bash scripts/status.sh -w` refreshes every 30 s and exits when
the machine goes quiet.

Every run writes `results/v1.1/<run_id>/run.log` itself, whatever launched it, so there is
always something to tail.

```bash
poetry run python scripts/90_report.py       # the results table, from every result.json
```

**Reporting a run back to me:** say *"d1 done"* — I read the artifacts off disk. If it
crashed, paste the traceback and leave the row alone.

---

## Scripts

| file | what |
|---|---|
| `scripts/00_build_cache.py` | decode + CQT caches. Run once per `--separate` variant. |
| `scripts/01_probe_representations.py` | frozen-representation probes + harness self-test |
| `scripts/10_train.py` | **every** Stage 1-4 experiment; stages are flags, not scripts |
| `scripts/90_report.py` | regenerates `results/v1.1/RESULTS.md` from all `result.json` |
| `scripts/status.sh` | **what is running, what is done, what is partial.** No arguments. |
| `scripts/run_batch1_cheap.sh` | Batch 1: cqt + resnet1d, Stages 1-2 |
| `scripts/run_batch2_hubert.sh` | Batch 2: distilHuBERT, Stages 1-2 |
| `scripts/run_batch3_sep_db.sh` | Batch 3: source separation + the DB prior |
| `colab/batch2_hubert.ipynb` | Batch 2 on a GPU |

---

## Queue

### Batch 0 — caches and cheap probes ✅

| id | what | cost | status |
|---|---|---|---|
| C0 | `00_build_cache.py` — 1960 clips to int16 + both CQT variants | 6 min | ✅ 2.34 GB, 0 errors |
| P0 | `01_probe_representations.py` — frozen probes, grouped 5-fold CV | 25 min | ✅ |

### Batch 1 — cqt + resnet1d, Stages 1-2 ✅

| id | what | val top-1 | status |
|---|---|---|---|
| c1 | CQT, fixed fmin | 0.111 | ✅ |
| c2 | CQT, Sa-anchored | **0.302** | ✅ |
| c2_shuffled | c2, tonics permuted *(control)* | 0.087 | ✅ control passes |
| r1 | jeevster ResNet, as-is | 0.146 | ✅ |
| r2n | jeevster ResNet, tonic-normalised audio | 0.287 | ✅ |
| r2c | jeevster ResNet, tonic by FiLM | 0.057 | ✅ |
| r2c_shuffled | r2c, conditioning tonic permuted *(control)* | 0.130 | ✅ control moves; FiLM path is wired |

An earlier c2_shuffled scoring 0.313 was void — cache-key bug, fixed, re-run. See `plan.md`.

### Batch 2 — distilHuBERT, Stages 1-2 ✅ *(Colab T4)*

| id | what | val top-1 | status |
|---|---|---|---|
| d1 | distilHuBERT, original recipe | 0.080 | ✅ |
| d2n | tonic-normalised audio | 0.087 | ✅ |
| d2c | tonic by FiLM | 0.076 | ✅ |
| d1_unfrozen | conv feature encoder unfrozen — GPU only | — | 🟨 optional; parked, may arrive 2026-09-01 |

**distilHuBERT is parked** — see `plan.md` for why. Stages 3-4 run on cqt and resnet1d only.

### Batch 3 — source separation and the DB prior 🔄

First pass 2026-08-31 21:07 (log `/tmp/batch3.out`): c3, r3, c4h completed; c4g, c4a, r4g
crashed at the end of epoch 0 on a `trainer.evaluate` bug — see `plan.md`. Fixed, smoke-
tested one epoch each, relaunched 23:10 (log `/tmp/batch3b.out`) for the three that failed.

```bash
nohup bash scripts/run_batch3_sep_db.sh > /tmp/batch3.out 2>&1 &
```

Builds the HPSS cache first (~20 min, one time), then runs six. All on top of Stage 2's
winner — audio-level tonic normalisation — since that is what Batch 1 settled.

| id | stage | what | status |
|---|---|---|---|
| c3 | 3 | CQT + Sa-anchor, over HPSS melody stem | ✅ 0.304 |
| r3 | 3 | resnet1d + normalised audio, over HPSS melody stem | ✅ 0.113 |
| c4g | 4 | CQT + Sa-anchor, graded label smoothing (`--graded-alpha 0.3`) | 🔄 |
| c4a | 4 | CQT + Sa-anchor, auxiliary swar-occupancy head (`--aux-weight 0.3`) | 🟨 queued |
| c4h | 4 | CQT + Sa-anchor, DB-template head — M12's mechanism, learned | ✅ **0.417** best so far |
| r4g | 4 | resnet1d + normalised audio, graded label smoothing | 🟨 queued |

---

## Conventions

**`run_id`** — the plan.md tag, lowercased, plus the variant: `c2`, `c2_shuffled`, `r4g`.
One directory per run under `results/v1.1/`.

**`result.json`** carries the same keys for every run, so runs compare without re-reading
code:

```json
{
  "run_id": "d1", "stage": 1, "arch": "distilhubert",
  "data_revision": "326caef0bc01da44ad46e4d9c65a5146da6bcc5b",
  "config": { "...": "every knob, verbatim" },
  "split": "grouped-val",
  "metrics": { "top1": 0.0, "top5": 0.0, "mrr": 0.0, "macro_f1": 0.0, "video_vote": 0.0 },
  "musical": { "mistake_affinity": 0.0, "...": 0.0 },
  "temperature": 1.0, "best_epoch": 0, "wall_clock_s": 0
}
```

**Test-split discipline.** Scoring the held-out 150 requires an explicit `--test` flag and
gets a line below. Nothing scores test by accident.

### Test-split evaluations so far

| date | run_id | method | test top-1 | notes |
|---|---|---|---|---|
| — | — | — | — | none yet |
