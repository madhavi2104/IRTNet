import os
import random
import json
import torch
import pandas as pd
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms
from transformers import CLIPProcessor, CLIPModel

# Define models to evaluate
model_names = [
    "resnet50",
    "densenet121",
    "efficientnet_b0",
    "vgg16",
    "mobilenet_v3_large",
    "alexnet",
    "clip_vit_g_14_laion2b"  # Custom identifier for CLIP model
]

# Load Hugging Face CLIP processor and model
def load_clip_model():
    model = CLIPModel.from_pretrained("laion/CLIP-ViT-g-14-laion2B-s12B-b42K")
    processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-g-14-laion2B-s12B-b42K")
    with open("data/Synsets/ImageNet_synsets.txt", "r") as f:
        class_labels = [line.strip() for line in f]
    inputs = processor(text=class_labels, return_tensors="pt", padding=True)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return model, processor, text_features, class_labels

# Load models
def load_model(name):
    if name == "clip_vit_g_14_laion2b":
        return load_clip_model()
    else:
        model = getattr(models, name)(pretrained=True)
        model.eval()
        return model

# Initialize models
models_dict = {}
processors_dict = {}
for name in model_names:
    if name == "clip_vit_g_14_laion2b":
        model, processor, text_features, class_labels = load_clip_model()
        models_dict[name] = model
        processors_dict[name] = {
            "processor": processor,
            "text_features": text_features,
            "class_labels": class_labels,
        }
    else:
        models_dict[name] = load_model(name)

# Load class labels for torchvision models
with open("data/Synsets/ImageNet_synsets.txt", "r") as f:
    torchvision_class_labels = [line.strip() for line in f]

# Define image transformation for torchvision models
torchvision_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_prediction(image_path, model, processor=None, text_features=None, class_labels=None):
    try:
        image = Image.open(image_path).convert("RGB")
        if processor and text_features is not None:
            inputs = processor(images=image, return_tensors="pt")
            with torch.no_grad():
                image_features = model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            similarities = image_features @ text_features.T
            predicted_idx = similarities.argmax(dim=-1).item()
            predicted_label = class_labels[predicted_idx]
        else:
            input_tensor = torchvision_transform(image).unsqueeze(0)
            with torch.no_grad():
                output = model(input_tensor)
                predicted_idx = output.argmax(dim=1).item()
                predicted_label = torchvision_class_labels[predicted_idx]
        return predicted_label
    except UnidentifiedImageError:
        print(f"Warning: Unable to process image {image_path}. Skipping...")
        return None
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def process_subset_hierarchical(dataset_dir, subset_size=100):
    print(f"Processing dataset in {dataset_dir} (Subset of {subset_size} images)...")

    # Get all image paths and their corresponding true labels (synsets)
    image_paths = []
    true_labels = []
    for synset in os.listdir(dataset_dir):
        synset_dir = os.path.join(dataset_dir, synset)
        if not os.path.isdir(synset_dir):
            continue
        for img_name in os.listdir(synset_dir):
            img_path = os.path.join(synset_dir, img_name)
            image_paths.append(img_path)
            true_labels.append(synset)

    # Subset images
    if subset_size < len(image_paths):
        sampled_indices = random.sample(range(len(image_paths)), subset_size)
        image_paths = [image_paths[i] for i in sampled_indices]
        true_labels = [true_labels[i] for i in sampled_indices]

    rows = []
    for img_path, true_label in zip(image_paths, true_labels):
        predictions = {}
        for model_name, model in models_dict.items():
            if model_name == "clip_vit_g_14_laion2b":
                processor = processors_dict[model_name]["processor"]
                text_features = processors_dict[model_name]["text_features"]
                class_labels = processors_dict[model_name]["class_labels"]
                predictions[model_name] = get_prediction(img_path, model, processor, text_features, class_labels)
            else:
                predictions[model_name] = get_prediction(img_path, model)

        if all(pred is None for pred in predictions.values()):
            print(f"Warning: No valid predictions for {img_path}. Skipping...")
            continue

        rows.append({
            "Item": os.path.relpath(img_path, dataset_dir),
            **{f"Answer {model_name}": predictions[model_name] for model_name in models_dict},
            "True Label": true_label
        })

    return pd.DataFrame(rows)

def main():
    subset_size = 100
    dataset_dir = "data/ImageNet/ILSVRC/Images"

    df = process_subset_hierarchical(dataset_dir, subset_size=subset_size)
    if df.empty:
        print(f"No data processed for {dataset_dir}. Skipping saving.")
        return
    output_csv = f"data/Processed/ImageNet_subset_results.csv"
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Dataset processed and saved to {output_csv}")

if __name__ == "__main__":
    main()

