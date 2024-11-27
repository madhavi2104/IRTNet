import os
import subprocess

# Define paths for the datasets
DATASET_DIR = "data/"
DATASETS = {
    "ImageNetR": "https://github.com/hendrycks/imagenet-r",
    "ImageNet_Sketch": "https://github.com/HaohanWang/ImageNet-Sketch",
    "ImageNet_V2": "https://github.com/modestyachts/ImageNetV2",
    "ImageNot": "https://github.com/hendrycks/imagenot",
}

def clone_repository(name, url):
    """Clone a dataset repository if not already present."""
    repo_path = os.path.join(DATASET_DIR, name)
    if not os.path.exists(repo_path):
        print(f"Cloning {name}...")
        subprocess.run(["git", "clone", url, repo_path], check=True)
    else:
        print(f"{name} already exists. Skipping.")

if __name__ == "__main__":
    # Create the dataset directory if it doesn't exist
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)

    # Clone each dataset repository
    for name, url in DATASETS.items():
        clone_repository(name, url)
    print("All datasets cloned successfully.")
