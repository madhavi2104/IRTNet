# import os
# import random
# import json
# import torch
# import pandas as pd
# from PIL import Image, UnidentifiedImageError
# from torchvision import models, transforms
# from transformers import CLIPProcessor, CLIPModel

# # Define models to evaluate
# model_names = [
#     "resnet50",
#     "densenet121",
#     "efficientnet_b0",
#     "vgg16",
#     "mobilenet_v3_large",
#     "alexnet",
#     "clip_vit_g_14_laion2b"  # Custom identifier for CLIP model
# ]

# # Load Hugging Face CLIP processor and model
# def load_clip_model():
#     """Load Hugging Face CLIP model and precompute class text embeddings."""
#     model = CLIPModel.from_pretrained("laion/CLIP-ViT-g-14-laion2B-s12B-b42K")
#     processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-g-14-laion2B-s12B-b42K")

#     # Load class labels (assuming ImageNet synsets are used)
#     with open("data/Synsets/ImageNet_synsets.txt", "r") as f:
#         class_labels = [line.strip() for line in f]

#     # Precompute text embeddings for class labels
#     inputs = processor(text=class_labels, return_tensors="pt", padding=True)
#     with torch.no_grad():
#         text_features = model.get_text_features(**inputs)
#     text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # Normalize embeddings

#     return model, processor, text_features, class_labels

# # Load models
# def load_model(name):
#     """Load pre-trained model based on its name."""
#     if name == "clip_vit_g_14_laion2b":
#         return load_clip_model()
#     else:
#         model = getattr(models, name)(pretrained=True)
#         model.eval()
#         return model

# # Initialize models
# models_dict = {}
# processors_dict = {}  # For storing CLIP-specific data
# for name in model_names:
#     if name == "clip_vit_g_14_laion2b":
#         model, processor, text_features, class_labels = load_clip_model()
#         models_dict[name] = model
#         processors_dict[name] = {
#             "processor": processor,
#             "text_features": text_features,
#             "class_labels": class_labels,
#         }
#     else:
#         models_dict[name] = load_model(name)

# # Load class labels for torchvision models
# with open("data/Synsets/ImageNet_synsets.txt", "r") as f:
#     torchvision_class_labels = [line.strip() for line in f]

# # Define image transformation for torchvision models
# torchvision_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# def get_prediction(image_path, model, processor=None, text_features=None, class_labels=None):
#     """Get model prediction for a single image."""
#     try:
#         image = Image.open(image_path).convert("RGB")
#         if processor and text_features is not None:  # Use CLIP model
#             inputs = processor(images=image, return_tensors="pt")
#             with torch.no_grad():
#                 image_features = model.get_image_features(**inputs)
#             image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # Normalize

#             # Compute similarities and find the best match
#             similarities = image_features @ text_features.T
#             predicted_idx = similarities.argmax(dim=-1).item()
#             predicted_label = class_labels[predicted_idx]
#         else:  # Use torchvision model
#             input_tensor = torchvision_transform(image).unsqueeze(0)  # Add batch dimension
#             with torch.no_grad():
#                 output = model(input_tensor)
#                 predicted_idx = output.argmax(dim=1).item()
#                 predicted_label = torchvision_class_labels[predicted_idx]  # Map index to label
#         return predicted_label
#     except UnidentifiedImageError:
#         print(f"Warning: Unable to process image {image_path}. Skipping...")
#         return None
#     except Exception as e:
#         print(f"Error processing {image_path}: {e}")
#         return None

# def get_sample_images(dataset_dir, subset_size):
#     """Randomly sample a subset of images from the dataset."""
#     image_paths = []
#     for root, _, files in os.walk(dataset_dir):
#         for file in files:
#             if file.endswith((".jpg", ".png", ".JPEG", ".jpeg")):
#                 image_paths.append(os.path.join(root, file))
    
#     if len(image_paths) <= subset_size:
#         print(f"Subset size {subset_size} exceeds total images {len(image_paths)}. Using all images.")
#         return image_paths
    
#     return random.sample(image_paths, subset_size)

# def process_subset(dataset_dir, synset_file, class_info_path, dataset_name, subset_size=100):
#     """Process a subset of a dataset and return a DataFrame."""
#     print(f"Processing {dataset_name} (Subset of {subset_size} images)...")

#     # Load synsets
#     try:
#         with open(synset_file, "r") as f:
#             valid_synsets = set(line.strip() for line in f)
#     except FileNotFoundError:
#         print(f"Error: Synset file {synset_file} not found. Skipping {dataset_name}.")
#         return pd.DataFrame()

#     # Load class_info.json for cid to wnid mapping (if provided)
#     label_to_synset = {}
#     if class_info_path:
#         try:
#             with open(class_info_path, 'r') as f:
#                 class_info = json.load(f)
#             label_to_synset = {str(item['cid']): item['wnid'] for item in class_info}
#         except FileNotFoundError:
#             print(f"Error: class_info.json file {class_info_path} not found. Skipping {dataset_name}.")
#             return pd.DataFrame()

#     subset_image_paths = get_sample_images(dataset_dir, subset_size)
#     rows = []

#     for image_path in subset_image_paths:
#         item_name = os.path.relpath(image_path, dataset_dir)
#         folder_label = os.path.basename(os.path.dirname(image_path))
#         true_label_synset = label_to_synset.get(folder_label) if class_info_path else folder_label

#         if not true_label_synset or true_label_synset not in valid_synsets:
#             print(f"Warning: True label {folder_label} not in synsets. Skipping {image_path}.")
#             continue

#         predictions = {}
#         for model_name, model in models_dict.items():
#             if model_name == "clip_vit_g_14_laion2b":
#                 processor = processors_dict[model_name]["processor"]
#                 text_features = processors_dict[model_name]["text_features"]
#                 class_labels = processors_dict[model_name]["class_labels"]
#                 predictions[model_name] = get_prediction(image_path, model, processor, text_features, class_labels)
#             else:
#                 predictions[model_name] = get_prediction(image_path, model)

#         if all(pred is None for pred in predictions.values()):
#             print(f"Warning: No valid predictions for {image_path}. Skipping...")
#             continue

#         rows.append({
#             "Item": item_name,
#             **{f"Answer {model_name}": predictions[model_name] for model_name in models_dict},
#             "True Label": true_label_synset
#         })

#     return pd.DataFrame(rows)

# def main():
#     subset_size = 100
#     datasets = [
#         {"name": "ImageNet", "dir": "data/ImageNet/ILSVRC/ILSVRC2012_img_val", "synset": "data/Synsets/ImageNet_synsets.txt", "class_info": None},
#         {"name": "ImageNet-Sketch", "dir": "data/ImageNet_Sketch/data/sketch", "synset": "data/Synsets/Sketch_synsets.txt", "class_info": None},
#         {"name": "ImageNet-V2", "dir": "data/ImageNet_V2/imagenetv2-matched-frequency-format-val", "synset": "data/Synsets/ImageNet_V2_synsets.txt", "class_info": "data/ImageNet_V2/class_info.json"},
#         {"name": "ImageNet-R", "dir": "data/ImageNetR/imagenet-r", "synset": "data/Synsets/ImageNet_R_synsets.txt", "class_info": None},
#         {"name": "ImageNot", "dir": "data/ImageNot", "synset": "data/Synsets/ImageNot_synsets.txt", "class_info": None},
#     ]

#     for dataset in datasets:
#         df = process_subset(dataset["dir"], dataset["synset"], dataset["class_info"], dataset["name"], subset_size=subset_size)
#         if df.empty:
#             print(f"No data processed for {dataset['name']}. Skipping saving.")
#             continue
#         output_csv = f"data/Processed/{dataset['name']}_subset_results.csv"
#         os.makedirs(os.path.dirname(output_csv), exist_ok=True)
#         df.to_csv(output_csv, index=False)
#         print(f"{dataset['name']} subset processed and saved to {output_csv}")

# if __name__ == "__main__":
#     main()


# Load ImageNet label mapping
with open(r"E:/Thesis/IRTNet/data/ImageNet/imagenet_class_index.json", "r") as f:
    imagenet_labels = json.load(f)

# Print the first few entries to verify the format
print(list(imagenet_labels.items())[:5])