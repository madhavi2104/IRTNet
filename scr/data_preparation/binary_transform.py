import pandas as pd
import os

# Define dataset filenames
datasets = {
    "ImageNet": "data\Processed\ImageNet_corrected_results_normalized.csv",
    "ImageNet-Sketch": "data/Processed/ImageNet-Sketch_corrected_results_normalized.csv",
    "ImageNet-V2": "data/Processed/ImageNet-V2_corrected_results_normalized.csv",
    "ImageNet-R": "data/Processed/ImageNet-R_corrected_results_normalized.csv"
}

# Output directory
output_dir = "data/Binary_Processed/"
os.makedirs(output_dir, exist_ok=True)

# Process each dataset
for name, file_path in datasets.items():
    print(f"Processing {name}...")
    
    # Load dataset
    df = pd.read_csv(file_path)
    
    # Identify model prediction columns
    models = [col for col in df.columns if col.startswith("Answer_")]
    
    # Convert predictions to binary responses (1 = correct, 0 = incorrect)
    binary_df = df.copy()
    for model in models:
        binary_df[model] = (df[model] == df["True Label"]).astype(int)

    # Keep only transformed binary responses
    binary_df = binary_df[models]

    # Save transformed dataset
    output_file = os.path.join(output_dir, f"Binary_Transformed_{name}.csv")
    binary_df.to_csv(output_file, index=False)
    
    print(f"Saved: {output_file}")

print("All datasets processed successfully!")
