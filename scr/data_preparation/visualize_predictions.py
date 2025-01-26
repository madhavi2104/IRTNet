import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy.io import loadmat

# Load meta.mat
meta_mat_path = r"E:\Thesis\IRTNet\data\ImageNet\ILSVRC\ILSVRC2012_devkit_t12\data\meta.mat"  # Update with your meta.mat file path
meta_mat = loadmat(meta_mat_path)

meta_synsets = meta_mat["synsets"]
meta_synset_to_label = {
    entry[1][0]: entry[2][0] for entry in meta_synsets
}

def get_class_name(class_id):
    """Convert synset ID to human-readable label."""
    return meta_synset_to_label.get(class_id, f"Unknown ({class_id})")

def visualize_predictions_with_top5(csv_file, dataset_dir, synsets_file):
    # Load predictions CSV
    df = pd.read_csv(csv_file)

    # Load synsets file for validation
    with open(synsets_file, "r") as f:
        valid_synsets = {line.strip() for line in f}

    for idx, row in df.iterrows():
        image_path = os.path.join(dataset_dir, row["Item"])
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue

        # Load image
        image = Image.open(image_path)

        # Prepare top-5 predictions
        predictions = {
            key.replace("Answer ", ""): row[key]
            for key in row.index if key.startswith("Answer ")
        }

        predictions_with_labels = {}
        for model, pred in predictions.items():
            label = get_class_name(pred)
            if pred not in valid_synsets:
                print(f"Warning: Prediction {pred} by {model} not in valid synsets.")
            predictions_with_labels[model] = f"{label} ({pred})"

        true_label = f"{get_class_name(row['True Label'])} ({row['True Label']})"

        # Display image with predictions
        plt.figure(figsize=(10, 8))
        plt.imshow(image)
        plt.axis("off")
        plt.title(
            f"True Label: {true_label}\n" +
            "\n".join([f"{model}: {label}" for model, label in predictions_with_labels.items()]),
            fontsize=12,
        )
        plt.show()

        input("Press Enter to view the next image (Ctrl+C to stop)...")

# Example usage
if __name__ == "__main__":
    csv_file = "data/Processed/ImageNet_subset_results.csv"  # Update with your CSV file path
    dataset_dir = "data/ImageNet/ILSVRC/Images"  # Update with your dataset path
    synsets_file = "data/Synsets/ImageNet_synsets.txt"  # Synsets file for validation
    visualize_predictions_with_top5(csv_file, dataset_dir, synsets_file)



