# Run registry

Bookkeeping only — **what is queued, what is running, what is done, how to drive it.**
Findings live in [`plan.md`](plan.md); numbers live in
[`results/v1.1/RESULTS.md`](results/v1.1/RESULTS.md). No scores in this file: a run is a
box that is empty, ticked, or on fire.

🟨 ready · 🔄 running · ✅ done · 🟥 not ready

🔄 means a process is *alive*. An interrupted batch reverts to 🟨 — partial epochs survive
in `state.pt` and are reused, but a run has produced nothing until `result.json` exists.

---

## Drive it

```bash
cd /Users/neerajaabhyankar/Repos/icm-shruti-analysis/raag-identifier/survey-aug-2026

bash scripts/status.sh                                             # what is running, now
nohup bash scripts/run_batch1_cheap.sh  > /tmp/batch1.out 2>&1 &   # launch a batch
poetry run python scripts/90_report.py --write                     # refresh RESULTS.md
```

`bash scripts/status.sh` needs no run id: it prints the live run with its current epoch,
best top-1 so far and the exact `tail -f`; then every run that has reported, ranked; then
anything part-finished. `-w` refreshes every 30 s and exits when the machine goes quiet.

`nohup … &` matters — in the foreground, closing the terminal kills the job.

| | |
|---|---|
| **Skipping** | a run with a `result.json` is skipped. To redo one, delete `results/v1.1/<id>/`; `FORCE=1` redoes all of them. |
| **Resuming** | re-running picks up from `state.pt` and rebuilds no cache it already has. |
| **Writing** | only under `results/v1.1/<run_id>/` and `cache/`. Never `push_to_hub`. |
| **Logging** | every run writes `results/v1.1/<run_id>/run.log` itself, however it was launched. |
| **Reporting back** | say *"d1 done"* — I read the artifacts off disk. If it crashed, paste the traceback and leave the row alone. |

---

## Scripts

| file | what |
|---|---|
| `scripts/00_build_cache.py` | decode + CQT caches. Once per `--separate` variant. |
| `scripts/01_probe_representations.py` | frozen-representation probes + harness self-test |
| `scripts/10_train.py` | **every** Stage 1–4 experiment; stages are flags, not scripts |
| `scripts/90_report.py` | `RESULTS.md`; `--notebook <ids>` for a `plan.md` table |
| `scripts/91_score_test.py` | scores the held-out 150 for finished runs, from `best.pt` |
| `scripts/status.sh` | what is running, done, partial. No arguments. |
| `scripts/run_batch1_cheap.sh` | Batch 1 — Stages 1–2, CQT + ResNet |
| `scripts/run_batch2_hubert.sh` | Batch 2 — Stages 1–2, distilHuBERT |
| `scripts/run_batch3_sep_db.sh` | Batch 3 — Stages 3–4 |
| `scripts/run_batch4_dbprior.sh` | Batch 4 — Stage 4 follow-ups + rigour |
| `scripts/run_batch5_seeds.sh` | Batch 5 — seed replication of the best run |
| `scripts/20_fuse_symbolic.py` | Batch 6 — Stage 5, fuse a DL run with a symbolic method |
| `scripts/21_melody_only.py` | the melody histogram alone, same split — Batch 7's control |
| `scripts/run_batch7_hybrid.sh` | Batch 7 — Stage 5, melody histogram as a model input |
| `colab/batch2_hubert.ipynb` | Batch 2 on a GPU |

---

## Queue

### Batch 0 — Stage 0 [harness] ✅
*local · 31 min*

| id | what | status |
|---|---|---|
| C0 | decode 1960 clips + both CQT variants | ✅ 2.34 GB, 0 errors |
| P0 | frozen-representation probes, grouped 5-fold CV | ✅ |

### Batch 1 — Stages 1–2 [baseline, tonic] on CQT + ResNet ✅
*local · ~4 h · `run_batch1_cheap.sh`*

| id | what | status |
|---|---|---|
| c1 | CQT, fixed fmin | ✅ |
| c2 | CQT, Sa-anchored | ✅ |
| c2_shuffled | c2, tonics permuted *(control)* | ✅ control passes |
| r1 | jeevster ResNet, as-is | ✅ |
| r2n | ResNet, tonic-normalised audio | ✅ |
| r2c | ResNet, tonic by FiLM | ✅ |
| r2c_shuffled | r2c, tonic permuted *(control)* | ✅ control moves; FiLM is wired |

An earlier c2_shuffled was void — cache-key bug, fixed, re-run.

### Batch 2 — Stages 1–2 [baseline, tonic] on distilHuBERT ✅
*Colab T4 · ~3 h · `colab/batch2_hubert.ipynb`*

| id | what | status |
|---|---|---|
| d1 | distilHuBERT, notebook recipe | ✅ |
| d2n | tonic-normalised audio | ✅ |
| d2c | tonic by FiLM | ✅ |
| d1_unfrozen | conv feature encoder unfrozen — GPU only | 🟨 optional; may arrive 2026-09-01 |

distilHuBERT is **parked**. Stages 3–4 ran on cqt and resnet1d only.

### Batch 3 — Stages 3–4 [separation, DB prior] ✅
*local · ~4 h incl. a one-time 20 min HPSS cache · `run_batch3_sep_db.sh`*

| id | what | status |
|---|---|---|
| c3 | c2 + HPSS melody stem | ✅ |
| r3 | r2n + HPSS melody stem | ✅ |
| c4g | c2 + graded label smoothing | ✅ |
| c4a | c2 + auxiliary occupancy head | ✅ |
| c4h | c2 + DB-template head | ✅ |
| r4g | r2n + graded label smoothing | ✅ |

c4g, c4a and r4g crashed on a `trainer.evaluate` bug in the first pass; fixed and re-run.

### Batch 4 — Stage 4 [DB prior] follow-ups + rigour ✅
*local · 6 h · `run_batch4_dbprior.sh`*

| id | what | status |
|---|---|---|
| dbprior_lam0 | learned templates, `--db-lam 0` — the ablation | ✅ |
| dbprior_36bins | 36 swar bins (~33 cents) | ✅ |
| dbprior_144bins | 144 swar bins | ✅ |
| dbprior_frozen | database templates frozen | ✅ |
| aug_jitter | c4h + pitch/gain jitter | ✅ best run in the project |
| seed1 | c4h at seed 1 | ✅ |
| seed2 | c4h at seed 2 | ✅ |
| cv5 | c4h at 5-fold CV | ✅ |

Two results changed earlier conclusions — the DB templates turned out not to matter, and the
seed spread on test is larger than most differences this survey reported. See `plan.md`.

### Batch 5 — seed replication of the best configuration ✅
*local · 77 min · `run_batch5_seeds.sh`*

| id | what | status |
|---|---|---|
| aug_seed1 | aug_jitter at seed 1 | ✅ |
| aug_seed2 | aug_jitter at seed 2 | ✅ |

Augmentation's apparent win was a lucky seed: +0.025 ± 0.018 val over three seeds, not
significant. See `plan.md`.

### Batch 6 — Stage 5 [hybrid] ✅
*local · ~10 min/fusion · `scripts/20_fuse_symbolic.py`*

No training: reads a DL run's `best.pt`, refits the symbolic method on the same 1350-clip
fit half, sweeps the mixing weight on val, applies it once to test.

```bash
poetry run python scripts/20_fuse_symbolic.py --dl aug_jitter --symbolic m14
```

| id | what | status |
|---|---|---|
| fuse_aug_jitter_m14 | aug_jitter + M14 | ✅ |
| fuse_aug_seed1_m14 | seed 1 | ✅ |
| fuse_aug_seed2_m14 | seed 2 | ✅ |
| fuse_c4h_m14 | c4h + M14, different DL parent | ✅ |

Best result in the project: fused test **0.447 ± 0.012** over three seeds, against
0.373 ± 0.031 for the DL model alone and 0.373-0.400 across the symbolic family.

### Release 09/01 ✅
*`../best-model-09-01/` · not a survey run*

The fusion, rebuilt as a standalone model: no imports outside its own directory, `pip
install -r requirements.txt`, `train.py` to reproduce it from the pinned dataset. Its
symbolic half is the naive histogram rather than M14 -- the same score at seed 0 without the
native `vamp` dependency (see `plan.md`).

Retrained on **all 1810 training clips** by its own script: **test top-1 0.487, top-5 0.820**
through the public inference path. The same script holding out a fifth reproduces the survey
at 0.447 (survey: 0.440), which is what says the rewrite is faithful.

| id | what | status |
|---|---|---|
| fuse_aug_jitter_melody | CQT + histogram fusion, seed 0 | ✅ |
| fuse_aug_seed1_melody / _seed2 | the same at seeds 1, 2 | ✅ |

### Batch 7 — Stage 5 [hybrid] as a model input ✅
*local · ~2 h · `run_batch7_hybrid.sh`*

Batch 6 averaged two finished models. This one hands the melody histogram to the network
as an input (`--melody`) and trains the two together, so the head can read both at once.

| id | what | status |
|---|---|---|
| melody_only | the histogram alone, logreg, same split *(control)* | ✅ |
| melody_only_seed1 / _seed2 | the same control at seeds 1, 2 | ✅ |
| hybrid_feat | aug_jitter + the histogram as an input | ✅ |
| hybrid_seed1 | hybrid_feat at seed 1 | ✅ |
| hybrid_seed2 | hybrid_feat at seed 2 | ✅ |
| hybrid_nodb | hybrid_feat without the DB-template head | ✅ |

Concatenation does not beat either branch on test; Batch 6's averaging still wins. The
histogram alone matches the symbolic champion. See `plan.md`.

Not queued, from that finding: modality dropout on the histogram, and a two-phase fit
(trunk first, joint head second) — both aimed at the shortcut the runs exposed.

---

## Conventions

**`run_id`** — the `plan.md` tag, lowercased, plus the variant: `c2`, `c2_shuffled`, `r4g`.
One directory per run under `results/v1.1/`.

**`result.json`** — the same keys for every run, so runs compare without re-reading code:

```json
{
  "run_id": "d1", "stage": 1, "arch": "distilhubert",
  "data_revision": "326caef0bc01da44ad46e4d9c65a5146da6bcc5b",
  "config": { "...": "every knob, verbatim" },
  "split": "grouped-val",
  "metrics": { "top1": 0.0, "top5": 0.0, "mrr": 0.0, "macro_f1": 0.0, "video_vote": 0.0 },
  "musical": { "mistake_affinity": 0.0, "...": 0.0 },
  "test":    { "metrics": {}, "musical": {} },
  "temperature": 1.0, "best_epoch": 0, "wall_clock_s": 0
}
```

**Test-split discipline** *(changed 2026-09-01)* — every run scores the held-out 150 at the
end and writes it into its own `result.json`; `--no-test` opts out. The training loop still
never sees test: splits are video-disjoint and checkpoint selection is on val top-1, so each
number is honest on its own. Choosing a method *by* its test score is the thing to avoid —
`plan.md` has the size of that bias.

Backfill for runs that predate the change:

```bash
poetry run python scripts/91_score_test.py                # every run missing a test score
poetry run python scripts/91_score_test.py c4h            # just one
poetry run python scripts/91_score_test.py --device cpu   # leave the GPU to a live batch
```
