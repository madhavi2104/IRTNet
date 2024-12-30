import os
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image, UnidentifiedImageError
from transformers import AutoProcessor, AutoModelForImageClassification
import pandas as pd

# Define Torchvision models
model_names = [
    "resnet50",
    "densenet121",
    "efficientnet_b0",
    "vgg16",
    "mobilenet_v3_large",
    "alexnet"
]

def load_model(name):
    """Load pre-trained Torchvision model."""
    model = getattr(models, name)(weights="DEFAULT")
    model.eval()
    return model

# Load all Torchvision models
models_dict = {name: load_model(name) for name in model_names}

# Define Hugging Face models
huggingface_model_names = [
    "google/vit-base-patch16-224-in21k",  # Example Hugging Face model 
    # laion/CoCa-ViT-L-14-laion2B-s13B-b90k
    # Add more Hugging Face model names here
]

def load_huggingface_model(name):
    """Load Hugging Face model and processor."""
    try:
        processor = AutoProcessor.from_pretrained(name)
        model = AutoModelForImageClassification.from_pretrained(name)
        model.eval()
        return processor, model
    except Exception as e:
        print(f"Error loading Hugging Face model {name}: {e}")
        return None, None

# Load Hugging Face models
huggingface_models = {
    name: load_huggingface_model(name) for name in huggingface_model_names
}

# Image transformation for Torchvision models
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_prediction(image_path, model):
    """Get prediction using a Torchvision model."""
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

def get_huggingface_prediction(image_path, processor, model):
    """Get prediction using a Hugging Face model."""
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_idx = outputs.logits.argmax(dim=1).item()
        return predicted_idx
    except Exception as e:
        print(f"Error processing {image_path} with Hugging Face model: {e}")
        return None

def process_dataset(dataset_dir, synset_file, dataset_name):
    """
    Process a dataset and return a DataFrame.
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
                    continue

                # Get predictions from all models
                predictions = {}

                # Torchvision models
                for model_name, model in models_dict.items():
                    predictions[f"Answer {model_name}"] = get_prediction(image_path, model)

                # Hugging Face models
                for model_name, (processor, model) in huggingface_models.items():
                    if processor and model:
                        predictions[f"Answer {model_name}"] = get_huggingface_prediction(
                            image_path, processor, model
                        )

                # Check if all predictions are None (image loading issue)
                if all(value is None for value in predictions.values()):
                    print(f"Warning: No valid predictions for {image_path}. Skipping...")
                    continue

                # Append row
                rows.append({
                    "Item": item_name,
                    **predictions,
                    "True Label": true_label,
                })

    # Convert to DataFrame
    return pd.DataFrame(rows)

def main():
    # Dataset configurations
    datasets = [
        {"name": "ImageNet", "dir": "data/ImageNet/ILSVRC/ILSVRC2012_img_val", "synset": "data/Synsets/ImageNet_synsets.txt"},
        {"name": "ImageNet-Sketch", "dir": "data/ImageNet_Sketch/data/sketch", "synset": "data/Synsets/ImageNet_Sketch_label_mapping.txt"},
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

# didnt work for google/vit-base-patch16-224-in21k -> all labels are 1
