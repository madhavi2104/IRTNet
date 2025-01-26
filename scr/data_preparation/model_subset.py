import os
import random
import json
import torch
import pandas as pd
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms
from torchvision.models import get_model_weights

# Define models to evaluate
model_names = [
    "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
    "densenet121", "densenet161", "densenet169", "densenet201",
    "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3", "efficientnet_b4",
    "vgg11", "vgg13", "vgg16", "vgg19",
    "mobilenet_v2", "mobilenet_v3_large", "mobilenet_v3_small",
    "alexnet", "shufflenet_v2_x0_5", "shufflenet_v2_x1_0",
    "squeezenet1_0", "squeezenet1_1", "inception_v3"
]

# Load Torchvision model weights for correct class labels
def load_torchvision_class_labels(model_name):
    try:
        # Retrieve weights and their associated categories
        weights_enum = get_model_weights(model_name)
        weights = weights_enum.DEFAULT
        categories = weights.meta["categories"]

        # Log success
        print(f"Loaded {len(categories)} categories for {model_name} from model metadata.")
        return categories
    except AttributeError:
        # Log a warning if weights or metadata are unavailable
        print(f"Warning: {model_name} does not provide metadata. Predictions may be inaccurate.")
        return []
    except Exception as e:
        print(f"Error loading class labels for {model_name}: {e}")
        return []

# Load Mapping File for Numeric Labels to Synsets: Handles imagenet_class_index.json
def load_class_index_mapping(mapping_file):
    with open(mapping_file, "r") as f:
        mapping = json.load(f)

    if not isinstance(mapping, dict):
        raise ValueError("Expected a dictionary format for imagenet_class_index.json.")

    # Reverse mapping: {wnid: synset}
    return {v[0]: v[1] for k, v in mapping.items()}

# Load Mapping File for Numeric Labels to Synsets: Handles class_info.json
def load_class_info_mapping(mapping_file):
    with open(mapping_file, "r") as f:
        mapping = json.load(f)

    if not isinstance(mapping, list):
        raise ValueError("Expected a list format for class_info.json.")
    
    # {cid: synset}
    return {entry["cid"]: entry["synset"][0] for entry in mapping}

# Wrapper to Load the Appropriate Mapping
def load_label_mapping(mapping_file, file_type):
    if file_type == "class_info":
        return load_class_info_mapping(mapping_file)
    elif file_type == "class_index":
        return load_class_index_mapping(mapping_file)
    else:
        raise ValueError(f"Unknown file type: {file_type}")

# Initialize models and processors
def load_model(name):
    try:
        # Dynamically retrieve the correct weights enum for the model
        weights_enum = get_model_weights(name)
        weights = weights_enum.DEFAULT  # Use the most up-to-date weights
        
        # Load the model with weights
        model = getattr(models, name)(weights=weights)
        model.eval()
        return model
    except AttributeError:
        print(f"Error: Model {name} is not available in torchvision.models.")
        return None
    except Exception as e:
        print(f"Error loading model {name}: {e}")
        return None

models_dict = {}
torchvision_class_labels_dict = {}

for name in model_names:
    model = load_model(name)
    if model:
        models_dict[name] = model
        torchvision_class_labels_dict[name] = load_torchvision_class_labels(name)

# Define image transformations for Torchvision models
torchvision_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Predict the label for a single image
def get_prediction(image_path, model, model_name):
    try:
        image = Image.open(image_path).convert("RGB")
        if model_name not in torchvision_class_labels_dict or not torchvision_class_labels_dict[model_name]:
            raise ValueError(f"Class labels not available for model: {model_name}")
        input_tensor = torchvision_transform(image).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
            predicted_idx = output.argmax(dim=1).item()
            predicted_label = torchvision_class_labels_dict[model_name][predicted_idx]
        return predicted_label
    except UnidentifiedImageError:
        print(f"Warning: Unable to process image {image_path}. Skipping...")
        return None
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# Generalized dataset processing
def process_dataset(dataset_dir, dataset_name, mapping_file=None, file_type=None, subset_size=None):
    print(f"Processing {dataset_name}...")

    # Load mapping file for True Labels
    label_mapping = load_label_mapping(mapping_file, file_type) if mapping_file else None

    image_paths = []
    true_labels = []
    missing_folders = set()  # Collect missing folders

    # Iterate through dataset folders
    for folder_name in os.listdir(dataset_dir):  # Folder name (e.g., "0", "1", ...)
        folder_path = os.path.join(dataset_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        # Handle numeric folder names specifically for ImageNet-V2
        if file_type == "class_info" and folder_name.isdigit():
            key = int(folder_name)  # Convert folder name to numeric ID
        else:
            key = folder_name  # Use folder name as-is for other datasets

        true_label = label_mapping.get(key, "UNKNOWN")
        if true_label == "UNKNOWN":
            missing_folders.add(folder_name)

        # Append true_label for each image in the folder
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            image_paths.append(img_path)
            true_labels.append(true_label)

    # Log missing folders
    if missing_folders:
        print("\nWarning: The following folders were not found in the mapping file:")
        print(", ".join(sorted(missing_folders)))

    # Subset sampling
    if subset_size and subset_size < len(image_paths):
        if len(image_paths) != len(true_labels):
            raise ValueError(
                f"Mismatch in lengths: image_paths ({len(image_paths)}) vs true_labels ({len(true_labels)})."
            )
        sampled_indices = random.sample(range(len(image_paths)), subset_size)
        image_paths = [image_paths[i] for i in sampled_indices]
        true_labels = [true_labels[i] for i in sampled_indices]

    rows = []
    for img_path, true_label in zip(image_paths, true_labels):
        predictions = {}
        for model_name, model in models_dict.items():
            pred = get_prediction(img_path, model, model_name)
            predictions[model_name] = pred

        if all(pred is None for pred in predictions.values()):
            print(f"Warning: No valid predictions for {img_path}. Skipping...")
            continue

        rows.append({
            "Item": os.path.relpath(img_path, dataset_dir),
            **{f"Answer_{model_name}": predictions[model_name] for model_name in models_dict},
            "True Label": true_label
        })

    return pd.DataFrame(rows)

def normalize_labels(csv_path):
    """
    Normalizes labels in the CSV for consistent matching.
    Args:
        csv_path (str): Path to the CSV file to process.
    """
    print(f"Normalizing labels in CSV: {csv_path}...")

    # Load the CSV
    df = pd.read_csv(csv_path)

    # Normalize True Label and Answer columns
    def normalize(label):
        if isinstance(label, str):
            return label.lower().replace("_", " ").strip()
        return label

    # Normalize True Label
    if "True Label" in df.columns:
        df["True Label"] = df["True Label"].apply(normalize)

    # Normalize all Answer columns
    for col in df.columns:
        if col.startswith("Answer "):
            df[col] = df[col].apply(normalize)

    # Save the updated CSV
    normalized_csv_path = csv_path.replace(".csv", "_normalized.csv")
    df.to_csv(normalized_csv_path, index=False)
    print(f"Normalized CSV saved to: {normalized_csv_path}")


# Main function
def main():
    datasets = [
        {
            "name": "ImageNet",
            "dir": "data/ImageNet/ILSVRC/Images",
            "mapping_file": "data/ImageNet/imagenet_class_index.json",
            "file_type": "class_index",
            "subset_size": None  
        },
        {
            "name": "ImageNet-V2",
            "dir": "data/ImageNet_V2/imagenetv2-matched-frequency-format-val",
            "mapping_file": "data/ImageNet_V2/class_info.json",
            "file_type": "class_info",
            "subset_size": None  
        },
        {
            "name": "ImageNet-R",
            "dir": "data/ImageNetR/imagenet-r/",
            "mapping_file": "data/ImageNet/imagenet_class_index.json",
            "file_type": "class_index",
            "subset_size": None  
        },
        {
            "name": "ImageNet-Sketch",
            "dir": "data/ImageNet_Sketch/data/sketch/",
            "mapping_file": "data/ImageNet/imagenet_class_index.json",
            "file_type": "class_index",
            "subset_size": None    
        }
    ]

    for dataset in datasets:
        df = process_dataset(
            dataset_dir=dataset["dir"],
            dataset_name=dataset["name"],
            mapping_file=dataset.get("mapping_file"),
            file_type=dataset.get("file_type"),
            subset_size=dataset.get("subset_size")
        )
        if df.empty:
            print(f"No data processed for {dataset['name']}. Skipping saving.")
            continue

        # Save the initial CSV
        output_csv = f"data/Processed/{dataset['name']}_corrected_results.csv"
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"{dataset['name']} processed and saved to {output_csv}")

        # Normalize labels in the CSV
        normalize_labels(output_csv)

if __name__ == "__main__":
    main()






