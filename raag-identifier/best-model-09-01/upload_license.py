from huggingface_hub import HfApi

REPO_ID = "neerajaabhyankar/cqt-histogram-hindustani-raag-small"

api = HfApi()

api.upload_file(
    path_or_fileobj="LICENSE",
    path_in_repo="LICENSE",
    repo_id=REPO_ID,
    repo_type="model",
)

api.upload_file(
    path_or_fileobj="README.md",
    path_in_repo="README.md",
    repo_id=REPO_ID,
    repo_type="model",
)

print("License and README updated.")
