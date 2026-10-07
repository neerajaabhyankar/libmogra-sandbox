For work in this folder,

- Please work in this project's poetry env.
- Please use poetry-installed libmogra (https://pypi.org/project/libmogra/0.4.3/).
- Use Mukhyanga (i.e. pakad) information from the RaagDB of the installed libmogra.
- Please use the dataset repo and revision pinned below. Please use only the training split for all learning and tuning purposes.
  ```
  DATASET_REPO_ID = "https://huggingface.co/datasets/neerajaabhyankar/hindustani-raag-small"
  DATASET_REVISION = "326caef0bc01da44ad46e4d9c65a5146da6bcc5b"
  ```
- REUSE AS MUCH CODE AS POSSIBLE. Prioritize importing from
  - ../raag-identifier/utils/
  - ../raag-identifier/melody-*
  - other folders in ../ ONLY IF YOU MUST, PLEASE CLEARLY STATE THIS AT THE TOP OF EACH FILE IF YOU DO
- Use Essentia for pitch tracking unless you have good reason to try others. (This folder exists to try others: Melodia stays the baseline every model is compared against.)
- KEEP CODE MODULARIZED, SUCCINCT, REUSABLE, AND HUMAN-READABLE
- For all hard-coded variables/values, please either separate out var definitions at the top of each file, or in a separate config file/json/template for this project.
- Please do not ramble, be concise. For both code and text.
- For all tasks you do, note down workflow in plan.md. Please use the following emojis in-line for status updates<br>
  🟨 ready · 🔄 running · ✅ done · 🟥 not ready <br>
  And update and clean-up the plan.md periodically. This should serve as a working doc for you and a project notebook for me or any human to understand.

General instructions carried over from `../pakad-matcher` (Neeraja, 2026-09/10):

- **Ask before kicking off any long run.** Ask before any download (package, checkpoint, dataset): name it, its source and its size. If disk runs low, say so; Neeraja will clean up.
- **Never commit.** Stage what is needed; Neeraja commits.
- The full audios (`../raag-identifier/hindustani-raag-fullaudios/`) are **read-only**.
- **Imports go one way: `pakad-matcher` → `transcriber`.** Nothing here imports from `../pakad-matcher` (it will import models from here). Callers pass audio, sample rate and times in; models hand tracks back.
- **One subfolder per model** (`models/<name>/`), with its own README: what it outputs, how it was adapted, what it cost.
- Inference input is **audio + its Sa**, never the raag. The raag DB and raag labels may train; they never infer.
- **Test once, after freeze.** Choose models and settings on training/validation only; score a test set once, with everything frozen. No choice is ever steered by a test number. Never ask Neeraja for more labels to make up for a protocol lapse.
- No recording straddles training and evaluation; augmentation does not excuse it.
- Neeraja's notation keeps notes the pitch track misses (tanpura over the voice, tapering volume): keep them, flag them, never filter labels down to what a pitch tracker sees.
- Define every term in the glossary (here: `plan.md` § Glossary, or `../pakad-matcher/DATA.md`). Metrics: no undefined jargon or acronyms.
- Do not share code with `/Users/neerajaabhyankar/Repos/mogra-app`; this repo is upstream R&D.
- Read annotator comments (notation notes, judgment comments) before trusting labels.
