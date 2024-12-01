import os

def map_labels(synset_file, mapping_output):
    """
    Reads synset files and writes a standardized mapping.

    Parameters:
    - synset_file: Path to the synset file for the dataset.
    - mapping_output: Path to save the label mapping.
    """
    if not os.path.exists(synset_file):
        raise FileNotFoundError(f"Synset file {synset_file} not found.")

    with open(synset_file, 'r') as infile, open(mapping_output, 'w') as outfile:
        for line in infile:
            synset = line.strip()
            outfile.write(f"{synset}\n")

if __name__ == "__main__":
    # Directory containing synset files
    SYNSETS_DIR = "data/Synsets"

    # Define synset files for all datasets
    synset_files = {
        "ImageNet": os.path.join(SYNSETS_DIR, "ImageNet_synsets.txt"),
        "ImageNet_R": os.path.join(SYNSETS_DIR, "ImageNet_R_synsets.txt"),
        "ImageNet_Sketch": os.path.join(SYNSETS_DIR, "Sketch_synsets.txt"),
        "ImageNet_V2": os.path.join(SYNSETS_DIR, "ImageNet_V2_synsets.txt"),
        "ImageNot": os.path.join(SYNSETS_DIR, "ImageNot_synsets.txt"),
    }

    # Generate label mappings for each dataset
    for dataset, synset_file in synset_files.items():
        try:
            output_path = os.path.join(SYNSETS_DIR, f"{dataset}_label_mapping.txt")
            map_labels(synset_file, output_path)
            print(f"Label mapping created for {dataset}: {output_path}")
        except FileNotFoundError as e:
            print(e)

