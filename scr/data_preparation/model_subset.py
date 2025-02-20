import os
import random
import json
import torch
import pandas as pd
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms
from torchvision.models import get_model_weights

model_names = [
    "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
    "densenet121", "densenet161", "densenet169", "densenet201",
    "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3", "efficientnet_b4",
    "vgg11", "vgg13", "vgg16", "vgg19",
    "mobilenet_v2", "mobilenet_v3_large", "mobilenet_v3_small",
    "alexnet", "shufflenet_v2_x0_5", "shufflenet_v2_x1_0",
    "squeezenet1_0", "squeezenet1_1", "inception_v3"
]
# 27 models in total from torchvision.models

def load_torchvision_class_labels(model_name):
    try:
        weights_enum = get_model_weights(model_name)
        weights = weights_enum.DEFAULT
        categories = weights.meta["categories"]

        print(f"Loaded {len(categories)} categories for {model_name} from model metadata.")
        return categories
    except AttributeError:
        print(f"Warning: {model_name} does not provide metadata. Predictions may be inaccurate.")
        return []
    except Exception as e:
        print(f"Error loading class labels for {model_name}: {e}")
        return []

def load_class_index_mapping(mapping_file):
    with open(mapping_file, "r") as f:
        mapping = json.load(f)
    return {v[0]: v[1] for k, v in mapping.items()}

def load_class_info_mapping(mapping_file):
    with open(mapping_file, "r") as f:
        mapping = json.load(f)
    return {entry["cid"]: entry["synset"][0] for entry in mapping}

def extract_truelabel_from_filename(filename):
    """
    Extracts the true label from an ImageNet-A file name.
    Expected format: number_truelabel_description
    """
    parts = filename.split("_")
    if len(parts) > 1:
        return parts[1]  # The second part is the true label
    return "UNKNOWN"

def load_label_mapping(mapping_file, file_type):
    if file_type == "class_info":
        return load_class_info_mapping(mapping_file)
    elif file_type == "class_index":
        return load_class_index_mapping(mapping_file)
    else:
        raise ValueError(f"Unknown file type: {file_type}")

def load_model(name):
    try:
        weights_enum = get_model_weights(name)
        weights = weights_enum.DEFAULT
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

torchvision_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

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

def process_dataset(dataset_dir, dataset_name, mapping_file=None, file_type=None, subset_size=None):
    print(f"Processing {dataset_name}...")
    
    image_paths = []
    true_labels = []
    
    for folder_name in os.listdir(dataset_dir):
        folder_path = os.path.join(dataset_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
        
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            true_label = extract_truelabel_from_filename(img_name) if dataset_name == "ImageNet-A" else folder_name
            image_paths.append(img_path)
            true_labels.append(true_label)
    
    if subset_size and subset_size < len(image_paths):
        sampled_indices = random.sample(range(len(image_paths)), subset_size)
        image_paths = [image_paths[i] for i in sampled_indices]
        true_labels = [true_labels[i] for i in sampled_indices]
    
    rows = []
    for img_path, true_label in zip(image_paths, true_labels):
        predictions = {model_name: get_prediction(img_path, model, model_name) for model_name, model in models_dict.items()}
        if all(pred is None for pred in predictions.values()):
            print(f"Warning: No valid predictions for {img_path}. Skipping...")
            continue
        rows.append({"Item": os.path.relpath(img_path, dataset_dir), **{f"Answer_{model_name}": pred for model_name, pred in predictions.items()}, "True Label": true_label})
    
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

def main():
    datasets = [
        # {"name": "ImageNet", "dir": "data/ImageNet/ILSVRC/Images", "mapping_file": "data/ImageNet/imagenet_class_index.json", "file_type": "class_index", "subset_size": None},
        # {"name": "ImageNet-V2", "dir": "data/ImageNet_V2/imagenetv2-matched-frequency-format-val", "mapping_file": "data/ImageNet_V2/class_info.json", "file_type": "class_info", "subset_size": None},
        # {"name": "ImageNet-R", "dir": "data/ImageNetR/imagenet-r/", "mapping_file": "data/ImageNet/imagenet_class_index.json", "file_type": "class_index", "subset_size": None},
        # {"name": "ImageNet-Sketch", "dir": "data/ImageNet_Sketch/data/sketch/", "mapping_file": "data/ImageNet/imagenet_class_index.json", "file_type": "class_index", "subset_size": None},
        {"name": "ImageNet-A", "dir": "data/imagenet-a/", "subset_size": None}
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
        output_csv = f"data/Processed/{dataset['name']}_corrected_results.csv"
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"{dataset['name']} processed and saved to {output_csv}")

        # Normalize labels in the CSV
        normalize_labels(output_csv)

if __name__ == "__main__":
    main()












