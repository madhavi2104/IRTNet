import os
import logging

logging.basicConfig(level=logging.INFO)

def clean_dataset(dataset_path):
    """
    Ensures all image files in the dataset path are accessible and valid.

    Parameters:
    - dataset_path: Path to the dataset directory.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")

    for root, _, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(root, file)
            if not file.lower().endswith(('.jpeg', '.jpg', '.png')):
                logging.warning(f"Non-image file detected and skipped: {file_path}")
                continue

            if not os.path.isfile(file_path):
                logging.error(f"Missing or corrupted file: {file_path}")

if __name__ == "__main__":
    from config import DATASETS
    for dataset_name, dataset_path in DATASETS.items():
        logging.info(f"Cleaning dataset: {dataset_name}")
        clean_dataset(dataset_path)
