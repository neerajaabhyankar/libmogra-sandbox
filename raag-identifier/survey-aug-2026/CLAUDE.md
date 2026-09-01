# survey-aug-2026 — standing instructions

Verbatim brief for this folder, from the conversation that started it (2026-08-31).

## What this folder is

Reproduce the **DL-based** raag-classification methods against the latest dataset revision,
`neerajaabhyankar/hindustani-raag-small@326caef0bc01da44ad46e4d9c65a5146da6bcc5b` (**v1.1**).
The two architectures already tried elsewhere in the repo:

1. `../distilhubert-finetuned-hindustani-raag-small/` — trained on **v0**, recipe lives in
   `../hindustani-raag-identifier.ipynb`.
2. `../hindustani-raag-classifier-resnet/` — trained on **v0**, jeevster 1D-ResNet backbone.
3. `../motif-classifier/` — the current champion, trained/tuned on **v1.1**, symbolic
   (melody + libmogra DB). This is the bar to beat: **0.400 test top-1**.

## Rules

- **Do not disrupt the old code.** Import from it freely; every new file lives here.
- **Everything local. Never push code or models to the Hub.** No `push_to_hub`, no
  `HfApi().upload_*`. Reading pinned dataset/model revisions is fine.
- Use the poetry env (`poetry env activate`, or `poetry run …` from the repo root).
- `plan.md` is the lab notebook: what was tried, what worked, what failed. Negative results
  are kept, not deleted. Update it as work happens, not at the end.
- `RUNS.md` is the run registry: every long job, its command, where it writes, and its
  status. It is the shared surface for "you run this overnight and tell me what happened".
- Use the following emojis to track runs: 🟨 = ready, not started; 🔄 = in progress; ✅ = done; 🟥 = not ready
- For all in-progress runs, make a note in RUNS.md itself as to how to inspect them.
- Code should be **modular and interpretable** — written for someone else downloading it.
  Reusable functions over copy-paste; `common/` holds anything used twice.
- Keep cleaning up RUNS.md and plans.md -- these should read as systematic notebooks, not as brain dumps.

## Method scope, in the priority the brief set

For **both** named architectures, at minimum:

1. Retrain as-is on v1.1.
2. **(important)** Add the `tonic_hz` input.
3. Source separation as pre-processing.
4. **(important)** Use the libmogra DB as a prior.
5. *(stretch, only if nothing here beats motif-classifier)* Hybrid: append the naive
   melody-only features to the DL architectures.

Agreed additions/decisions:

- **A third architecture**: a small 2D ResNet over a **CQT rolled so Sa sits at bin 0** —
  exactly tonic-invariant, and its feature space is the same 12-bin space as the libmogra
  templates, so the DB prior plugs in natively rather than being bolted on.
- **Full 20 s clips**, not 5 s crops.
- **Test discipline: single-shot.** All tuning on video-grouped CV/val over the 1810 train
  clips; the 150 test clips are scored once per method, at the end.
- **Compute is hybrid**: cheap probes run locally for signal; anything long is staged as a
  bash script under `scripts/` for the user to trigger overnight, or as a Colab notebook
  under `colab/` when it does not fit on the M1. Colab has crashed on preprocessing steps
  before — so notebooks must checkpoint and resume rather than redo work.
