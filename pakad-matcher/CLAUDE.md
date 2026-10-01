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
- Use Essentia for pitch tracking unless you have good reason to try others.
- KEEP CODE MODULARIZED, SUCCINCT, REUSABLE, AND HUMAN-READABLE
- For all hard-coded variables/values, please either separate out var definitions at the top of each file, or in a separate config file/json/template for this project.
- Please do not ramble, be concise. For both code and text.
- For all tasks you do, note down workflow in plan.md. Please use the following emojis in-line for status updates<br>
  🟨 ready · 🔄 running · ✅ done · 🟥 not ready <br>
  And update and clean-up the plan.md periodically. This should serve as a working doc for you and a project notebook for me or any human to understand.

