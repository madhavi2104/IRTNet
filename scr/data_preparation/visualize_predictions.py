import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import json

# Load ImageNet label mapping
with open(r"E:/Thesis/IRTNet/data/ImageNet/imagenet_class_index.json", "r") as f:
    imagenet_labels = json.load(f)

# # Print the first few entries to verify the format
# print(list(imagenet_labels.items())[:5])

# [('0', ['n01440764', 'tench']), ('1', ['n01443537', 'goldfish']), ('2', ['n01484850', 'great_white_shark']), ('3', ['n01491361', 'tiger_shark']), ('4', ['n01494475', 'hammerhead'])]

def get_class_name(class_id):
    """Convert ImageNet class ID to a human-readable label."""
    for key, value in imagenet_labels.items():
        if value[0] == class_id:  # Compare the ImageNet ID (e.g., 'n01440764') to class_id
            return value[1].replace("_", " ")
    return class_id  # Return the class_id if not found

# # Test the function

get_class_name('1')


def visualize_predictions_with_top5(csv_file, dataset_dir):
    df = pd.read_csv(csv_file)

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
        predictions_with_labels = {
            model: f"{get_class_name(pred)} ({pred})"
            for model, pred in predictions.items()
        }

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
    visualize_predictions_with_top5(csv_file, dataset_dir)
