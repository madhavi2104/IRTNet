import os
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image, UnidentifiedImageError
import pandas as pd

# Define models to evaluate, including AlexNet
model_names = [
    "resnet50",
    "densenet121",
    "efficientnet_b0",
    "vgg16",
    "mobilenet_v3_large",
    "alexnet"
]

def load_model(name):
    """Load pre-trained PyTorch model."""
    model = getattr(models, name)(pretrained=True)
    model.eval()
    return model

# Load all models
models_dict = {name: load_model(name) for name in model_names}

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_prediction(image_path, model):
    """Get model prediction for a single image."""
    try:
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            output = model(input_tensor)
            predicted_idx = output.argmax(dim=1).item()
        return predicted_idx
    except UnidentifiedImageError:
        print(f"Warning: Unable to process image {image_path}. Skipping...")
        return None
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def process_dataset(dataset_dir, synset_file, dataset_name):
    """
    Process a dataset and return a DataFrame.
    - dataset_dir: Directory containing the dataset images.
    - synset_file: Path to the file containing synsets (class mappings).
    - dataset_name: Name of the dataset (for logging).
    """
    print(f"Processing {dataset_name}...")

    # Load synsets
    try:
        with open(synset_file, "r") as f:
            synsets = [line.strip() for line in f]
    except FileNotFoundError:
        print(f"Error: Synset file {synset_file} not found. Skipping {dataset_name}.")
        return pd.DataFrame()

    rows = []
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith((".jpg", ".png")):
                image_path = os.path.join(root, file)
                item_name = os.path.relpath(image_path, dataset_dir)  # Relative path as Item

                # Get true label based on folder name
                true_label = os.path.basename(os.path.dirname(image_path))
                if true_label not in synsets:
                    print(f"Warning: True label {true_label} not in synsets. Skipping {image_path}.")
                    continue  # Skip if the label is invalid

                # Get predictions from all models
                predictions = {
                    model_name: get_prediction(image_path, model)
                    for model_name, model in models_dict.items()
                }

                # Check if all predictions are None (image loading issue)
                if all(pred is None for pred in predictions.values()):
                    print(f"Warning: No valid predictions for {image_path}. Skipping...")
                    continue

                # Append row
                rows.append({
                    "Item": item_name,
                    **{f"Answer {model_name}": predictions[model_name] for model_name in models_dict},
                    "True Label": true_label
                })

    # Convert to DataFrame
    df = pd.DataFrame(rows)
    return df

def main():
    # Dataset configurations
    datasets = [
        {"name": "ImageNet", "dir": "data\ImageNet\ILSVRC\ILSVRC2012_img_val", "synset": "data/Synsets/ImageNet_synsets.txt"},
        {"name": "ImageNet-Sketch", "dir": "data/ImageNet_Sketch/data/sketch", "synset": "data/Synsets/Sketch_synsets.txt"},
        {"name": "ImageNet-V2", "dir": "data/ImageNet_V2/imagenetv2-matched-frequency-format-val", "synset": "data/Synsets/ImageNet_V2_synsets.txt"},
        {"name": "ImageNet-R", "dir": "data/ImageNetR/imagenet-r", "synset": "data/Synsets/ImageNet_R_synsets.txt"},
        {"name": "ImageNot", "dir": "data/ImageNot", "synset": "data/Synsets/ImageNot_synsets.txt"}
    ]

    for dataset in datasets:
        df = process_dataset(dataset["dir"], dataset["synset"], dataset["name"])
        if df.empty:
            print(f"No data processed for {dataset['name']}. Skipping saving.")
            continue
        output_csv = f"data/Processed/{dataset['name']}_results.csv"
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"{dataset['name']} processed and saved to {output_csv}")

if __name__ == "__main__":
    main()


 