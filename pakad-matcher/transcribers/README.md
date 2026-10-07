# transcribers/ -- ../transcriber models as pakad-matcher's pitch source

Kept apart from the Melodia work so far (`results/`). The models live in `../transcriber`
(its own `plan.md`); this folder only evaluates them here, with the unchanged scripts.

**How.** `config.PITCH_SOURCE` (env `PAKAD_PITCH_SOURCE`, default `melodia`) decides what
`fullaudio.contour()` returns, so the reader fit, phrase matcher and insights read the model's
track instead of Melodia's -- on the same frame grid -- and write to `<source>/results/`.

| file | role |
|---|---|
| `source.py` | the only import of `../transcriber`: cached tracks → Melodia's frame grid |
| `segments.py` | the time ranges a source must cover (notation, validation, insight train+val; test only for the final pick) |
| `run.py` | transcribe → reader refit → phrase val → insight val, logs in `<source>/logs/` |
| `<source>/results/` | that source's `reader.json`, `s7_choice.json`, `insights/` -- same names as `results/` |

Shared across sources (`results/`, `config.SHARED_RESULTS_DIR`): the phrase catalogue and the
frozen splits manifest. Frames outside the covered ranges read as unvoiced.

**Protocol.** Every source is compared with Melodia on training/validation numbers only. The
final pick alone is scored on test, once (`run.py --steps test`).
