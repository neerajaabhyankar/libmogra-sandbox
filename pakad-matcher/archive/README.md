# archive/: scripts that produced a recorded result and are no longer on the live path

Kept for provenance, not deleted. Every one is described in `plan.md` under its stage. Moved here
on 2026-10-03 (review cleanup). They import modules from the project root, so run them from there:
`PYTHONPATH=. poetry run python archive/<script>.py`. Some depend on APIs that have since changed;
to reproduce one exactly, check out the commit it last ran at (the code as of commit `cce7820` has
every one of them in place).

| script | stage | what it showed | result files |
|---|---|---|---|
| `run_s1.py`, `plot.py` | S1 | eyeball run of the first matcher | `results/s1/` |
| `run_s2.py` | S2 | the label-free gate (shuffles, other raags) failed, and was the wrong gate | `results/s2/` |
| `s3.py` | S3 | the first terminal annotation loop (pool v1) | `annotations/labels_pool1.jsonl` |
| `s4.py`, `features.py` | S4 | features built from annotator comments: negative | -- |
| `tune.py` | S4b | coordinate ascent of the matcher costs on the first 168 judgments -> the "(tuned)" keys of `config.MATCH` | -- |
| `s5a.py` | S5a | likelihood-ratio scoring: negative | -- |
| `s6.py` | S6 | scoring the free reading against notation: over-segmentation is the wall | `logs/` |
| `ctc_reader.py` | S12 | a learned (GRU) reader: held-out misread 0.618 vs 0.559, not adopted | -- |
| `insights_fit.py` | I1/I2 | insight rules tuned on notation (swar before a notated pause; direction note length) -- superseded by tuning on the insight clips | `results/insights/fit.json` |

**Do not rerun `tune.py`, `s4.py` or `s5a.py` as they are.** They read every judgment in
`annotations/labels.jsonl` directly, not through `audit.splits()`, so today they would fit on
test1 labels too. (When they ran, the labels were the 168 round-1 judgments, all now validation.)
The live code reads judgments only through `audit.judgments()` / `audit.splits()`.
