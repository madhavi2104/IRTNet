import os
import random
import shutil

def split_dataset(dataset_dir, output_dir, split_ratios=(0.7, 0.2, 0.1)):
    """
    Splits dataset into train, validation, and test sets.

    Parameters:
    - dataset_dir: Path to the dataset directory.
    - output_dir: Path to save the split datasets.
    - split_ratios: Tuple of train, validation, and test ratios.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for synset in os.listdir(dataset_dir):
        synset_dir = os.path.join(dataset_dir, synset)
        if not os.path.isdir(synset_dir):
            continue

        images = os.listdir(synset_dir)
        random.shuffle(images)

        train_split = int(len(images) * split_ratios[0])
        val_split = int(len(images) * (split_ratios[0] + split_ratios[1]))

        splits = {
            "train": images[:train_split],
            "val": images[train_split:val_split],
            "test": images[val_split:],
        }

        for split, files in splits.items():
            split_dir = os.path.join(output_dir, split, synset)
            os.makedirs(split_dir, exist_ok=True)
            for file in files:
                shutil.copy(os.path.join(synset_dir, file), os.path.join(split_dir, file))

if __name__ == "__main__":
    from config import DATASETS

    for dataset, path in DATASETS.items():
        output_dir = os.path.join(path, "Splits")
        split_dataset(path, output_dir)
