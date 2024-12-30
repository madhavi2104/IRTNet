import os
import random
import json
import torch
import pandas as pd
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms

# Define models to evaluate
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

# Load all models into a dictionary
models_dict = {name: load_model(name) for name in model_names}

# Define image transformation
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

def get_sample_images(dataset_dir, subset_size):
    """
    Randomly sample a subset of images from the dataset.
    - dataset_dir: Directory containing the dataset images.
    - subset_size: Number of images to include in the subset.
    """
    image_paths = []
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith((".jpg", ".png", ".JPEG", ".jpeg")):
                image_paths.append(os.path.join(root, file))
    
    if len(image_paths) <= subset_size:
        print(f"Subset size {subset_size} exceeds total images {len(image_paths)}. Using all images.")
        return image_paths
    
    # Randomly sample the subset
    return random.sample(image_paths, subset_size)

def process_subset(dataset_dir, synset_file, class_info_path, dataset_name, subset_size=100):
    """
    Process a subset of a dataset and return a DataFrame.
    - dataset_dir: Directory containing the dataset images.
    - synset_file: Path to the file containing synsets (class mappings).
    - class_info_path: Path to the class_info.json for mapping numerical labels (cid) to synsets (wnid).
    - dataset_name: Name of the dataset (for logging).
    - subset_size: Number of images to include in the subset.
    """
    print(f"Processing {dataset_name} (Subset of {subset_size} images)...")

    # Load synsets
    try:
        with open(synset_file, "r") as f:
            valid_synsets = set(line.strip() for line in f)
    except FileNotFoundError:
        print(f"Error: Synset file {synset_file} not found. Skipping {dataset_name}.")
        return pd.DataFrame()

    # Load class_info.json for cid to wnid mapping (if provided)
    label_to_synset = {}
    if class_info_path:
        try:
            with open(class_info_path, 'r') as f:
                class_info = json.load(f)
            label_to_synset = {str(item['cid']): item['wnid'] for item in class_info}
        except FileNotFoundError:
            print(f"Error: class_info.json file {class_info_path} not found. Skipping {dataset_name}.")
            return pd.DataFrame()

    # Get a random subset of image paths
    subset_image_paths = get_sample_images(dataset_dir, subset_size)

    rows = []
    for image_path in subset_image_paths:
        item_name = os.path.relpath(image_path, dataset_dir)  # Relative path as Item

        # Get true label based on folder name and map it to synset
        folder_label = os.path.basename(os.path.dirname(image_path))
        true_label_synset = label_to_synset.get(folder_label) if class_info_path else folder_label

        if not true_label_synset or true_label_synset not in valid_synsets:
            print(f"Warning: True label {folder_label} not in synsets. Skipping {image_path}.")
            continue

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
            "True Label": true_label_synset
        })

    # Convert to DataFrame
    df = pd.DataFrame(rows)
    return df

def main():
    # Dataset configurations for subsets
    subset_size = 100  # Adjust as needed
    datasets = [
        {
            "name": "ImageNet",
            "dir": "data/ImageNet/ILSVRC/ILSVRC2012_img_val",
            "synset": "data/Synsets/ImageNet_synsets.txt",
            "class_info": None  # No class_info.json needed for ImageNet
        },
        {
            "name": "ImageNet-Sketch",
            "dir": "data/ImageNet_Sketch/data/sketch",
            "synset": "data/Synsets/Sketch_synsets.txt",
            "class_info": None  # No class_info.json needed for Sketch
        },
        {
            "name": "ImageNet-V2",
            "dir": "data/ImageNet_V2/imagenetv2-matched-frequency-format-val",
            "synset": "data/Synsets/ImageNet_V2_synsets.txt",
            "class_info": "data/ImageNet_V2/class_info.json"  # Class info for ImageNet-V2
        },
        {
            "name": "ImageNet-R",
            "dir": "data/ImageNetR/imagenet-r",
            "synset": "data/Synsets/ImageNet_R_synsets.txt",
            "class_info": None  # No class_info.json needed for ImageNet-R
        },
        {
            "name": "ImageNot",
            "dir": "data/ImageNot",
            "synset": "data/Synsets/ImageNot_synsets.txt",
            "class_info": None  # No class_info.json needed for ImageNot
        }
    ]

    for dataset in datasets:
        df = process_subset(dataset["dir"], dataset["synset"], dataset["class_info"], dataset["name"], subset_size=subset_size)
        if df.empty:
            print(f"No data processed for {dataset['name']}. Skipping saving.")
            continue
        output_csv = f"data/Processed/{dataset['name']}_subset_results.csv"
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"{dataset['name']} subset processed and saved to {output_csv}")

if __name__ == "__main__":
    main()

