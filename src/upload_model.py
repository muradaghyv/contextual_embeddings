from huggingface_hub import HfApi
import os

local_model_path = "../test_encoder_only_m3_bge-m3_sd"

repo_id = "muradaghyv/finetuned_bge_m3_v1"

all_files = [os.path.join(dirpath, filename) for dirpath, _, filenames in os.walk(local_model_path) for filename in filenames]
api = HfApi()

print(f"Uploading model to {repo_id}. . .")

for file in all_files:
    path_in_repo = os.path.relpath(file, local_model_path)

    print(f"Uploading {file} to {path_in_repo}")
    api.upload_file(
        path_or_fileobj=file,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="model"
    )

print("Model upload completed successfully!")