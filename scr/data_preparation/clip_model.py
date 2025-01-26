import os
import random
import torch
import pandas as pd
from PIL import Image, UnidentifiedImageError
from transformers import CLIPProcessor, CLIPModel
import matplotlib.pyplot as plt

# Load CLIP Model and Processor
def load_clip_model():
    print("Loading CLIP model and processor...")
    model = CLIPModel.from_pretrained("laion/CLIP-ViT-g-14-laion2B-s12B-b42K")
    processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-g-14-laion2B-s12B-b42K")
    
    # Load synsets from ImageNet_synsets.txt
    with open("data/Synsets/ImageNet_synsets.txt", "r") as f:
        synsets = [line.strip() for line in f]
    
    # Generate enhanced prompts using synsets
    class_labels = [f"a realistic image of category {synset}" for synset in synsets]
    print(f"Sample prompts: {class_labels[:5]}")  # Debugging: Log sample prompts
    
    # Create text features for all class labels
    inputs = processor(text=class_labels, return_tensors="pt", padding=True)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    print(f"Loaded {len(class_labels)} class labels.")
    return model, processor, text_features, class_labels

# Preprocess and Predict
def predict_clip(image_path, model, processor, text_features, class_labels, true_label=None):
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Compute similarities
        similarities = image_features @ text_features.T
        predicted_idx = similarities.argmax(dim=-1).item()
        predicted_label = class_labels[predicted_idx]

        # Log similarity scores for debugging
        print(f"\nImage: {image_path}")
        if true_label:
            print(f"True Label: {true_label}")
        print(f"Predicted Label: {predicted_label}")
        print(f"Top-1 Similarity Score: {similarities.max().item()}")

        return predicted_label, similarities.max().item()
    except UnidentifiedImageError:
        print(f"Warning: Unable to process image {image_path}. Skipping...")
        return None, None
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, None

# Process Dataset
def process_clip_dataset(dataset_dir, subset_size=None):
    print("Processing dataset for CLIP...")
    model, processor, text_features, class_labels = load_clip_model()

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

    # Subset sampling
    if subset_size and subset_size < len(image_paths):
        sampled_indices = random.sample(range(len(image_paths)), subset_size)
        image_paths = [image_paths[i] for i in sampled_indices]
        true_labels = [true_labels[i] for i in sampled_indices]

    rows = []
    for img_path, true_label in zip(image_paths, true_labels):
        prediction, similarity = predict_clip(img_path, model, processor, text_features, class_labels, true_label=true_label)
        if prediction is None:
            continue
        rows.append({
            "Image": img_path,
            "Prediction": prediction,
            "True Label": true_label,
            "Similarity": similarity
        })

    return pd.DataFrame(rows)

# Visualization Function
def visualize_predictions(df):
    print("\nVisualizing Prediction Summary...")
    # Calculate counts
    correct = len(df[df["Prediction"] == df["True Label"]])
    incorrect = len(df[df["Prediction"] != df["True Label"]])
    total = correct + incorrect

    # Display metrics
    print(f"Total Images: {total}")
    print(f"Correctly Classified: {correct}")
    print(f"Misclassified: {incorrect}")

    # Create bar chart
    plt.figure(figsize=(8, 6))
    plt.bar(["Correct", "Misclassified"], [correct, incorrect], color=["green", "red"])
    plt.xlabel("Classification Result")
    plt.ylabel("Number of Images")
    plt.title("Prediction Summary: Correct vs. Misclassified")
    plt.show()

# Main Function
def main():
    dataset_dir = "data/ImageNet/ILSVRC/Images"  # Update this path for your dataset
    subset_size = 100  # Adjust subset size as needed
    df = process_clip_dataset(dataset_dir, subset_size=subset_size)
    if not df.empty:
        output_csv = "data/Processed/CLIP_results.csv"
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")
        
        # Visualize predictions
        visualize_predictions(df)
    else:
        print("No data processed. Check dataset or CLIP model configuration.")

if __name__ == "__main__":
    main()






