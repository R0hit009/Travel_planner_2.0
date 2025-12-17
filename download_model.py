from huggingface_hub import snapshot_download

# Download the model into your chosen folder
local_model_path = snapshot_download(
    repo_id="google/gemma-2-2b-it",   # change if using quantized version
    local_dir="C:/project/travel_ai/models/gemma-2-2b-it",  # your custom path
    local_dir_use_symlinks=False  # ensure real copy, not symlink
)

print("Model saved at:", local_model_path)
