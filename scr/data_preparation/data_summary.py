import os
from collections import Counter

def summarize_dataset(dataset_dir):
    """
    Summarizes dataset statistics.

    Parameters:
    - dataset_dir: Path to the dataset directory.
    """
    class_counts = Counter()
    for root, dirs, _ in os.walk(dataset_dir):
        for folder in dirs:
            folder_path = os.path.join(root, folder)
            num_files = len(os.listdir(folder_path))
            class_counts[folder] += num_files

    print("Dataset Summary:")
    for cls, count in class_counts.items():
        print(f"Class: {cls}, Number of images: {count}")

if __name__ == "__main__":
    from config import DATASETS

    for dataset, path in DATASETS.items():
        print(f"Summary for {dataset}:")
        summarize_dataset(path)
