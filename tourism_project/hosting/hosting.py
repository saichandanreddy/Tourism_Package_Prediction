from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import os

api = HfApi(token=os.getenv("HF_TOKEN"))

repo_id = "saichandanreddy/Tourism-Package-Prediction" # The target repository ID (space name)
repo_type = "space" # It's a Space, not a dataset or model

# Check if the space exists, if not, create it
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Hugging Face Space '{repo_id}' already exists. Proceeding with upload.")
except RepositoryNotFoundError:
    print(f"Hugging Face Space '{repo_id}' not found. Creating a new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Hugging Face Space '{repo_id}' created successfully.")

# Upload the deployment folder content to the Hugging Face Space
api.upload_folder(
    folder_path="tourism_project/deployment",     # the local folder containing your files
    repo_id=repo_id,                              # the target repo
    repo_type=repo_type,                          # dataset, model, or space
    path_in_repo="",                              # optional: subfolder path inside the repo
)

print(f"Successfully uploaded deployment files to Hugging Face Space '{repo_id}'.")
print(f"You can view your space at: https://huggingface.co/spaces/{repo_id}")
