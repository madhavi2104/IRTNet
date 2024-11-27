import scipy.io
import os

def extract_imagenet_synsets(meta_file, ground_truth_file, output_txt):
    """
    Extracts synsets for the ImageNet validation dataset.

    Parameters:
    - meta_file: Path to meta.mat containing synset-class index mapping.
    - ground_truth_file: Path to ILSVRC2012_validation_ground_truth.txt.
    - output_txt: Path to save the extracted synsets list.
    """
    # Load meta.mat to get the mapping of synsets to class indices
    meta = scipy.io.loadmat(meta_file)
    synsets = [item[0][0] for item in meta['synsets']['WNID']]

    # Load ground truth file
    with open(ground_truth_file, 'r') as f:
        ground_truth_indices = [int(line.strip()) for line in f]

    # Map ground truth indices to synsets
    ground_truth_synsets = [synsets[index - 1] for index in ground_truth_indices]

    # Save the unique synsets to a text file
    with open(output_txt, 'w') as f:
        for synset in sorted(set(ground_truth_synsets)):
            f.write(synset + '\n')

    print(f"Synsets saved to {output_txt}")

# Example usage:
meta_file_path = r"E:\Thesis\IRTNet\data\ImageNet\ILSVRC\Annotations\CLS-LOC\val\ILSVRC2012_devkit_t12\data\meta.mat"
ground_truth_file_path = r"E:\Thesis\IRTNet\data\ImageNet\ILSVRC\Annotations\CLS-LOC\val\ILSVRC2012_devkit_t12\data\ILSVRC2012_validation_ground_truth.txt"
output_file_path = r"E:\Thesis\IRTNet\data\Synsets\ImageNet_synsets.txt"

extract_imagenet_synsets(meta_file_path, ground_truth_file_path, output_file_path)


import os

def extract_synsets_from_sketch(data_dir, output_txt):
    """
    Extracts synsets from the ImageNet-Sketch dataset folder structure.

    Parameters:
    - data_dir: Path to the root folder containing ImageNet-Sketch data.
    - output_txt: Path to save the extracted synsets list.
    """
    synsets = set()

    # List directories in the data folder
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path):  # Synsets are folder names
            synsets.add(item)

    # Save the unique synsets to a text file
    with open(output_txt, 'w') as f:
        for synset in sorted(synsets):
            f.write(synset + '\n')

    print(f"Synsets saved to {output_txt}")

# Example usage:
data_dir_path = r"E:\Thesis\IRTNet\data\ImageNet_Sketch\data\sketch"
output_file_path = r"E:\Thesis\IRTNet\data\Synsets\Sketch_synsets.txt"

extract_synsets_from_sketch(data_dir_path, output_file_path)

import json

# Path to the class_info.json file
class_info_path = r"E:\Thesis\IRTNet\data\ImageNet_V2\class_info.json"  # Adjust the path as needed

# Load the class_info.json file
with open(class_info_path, 'r') as f:
    class_info = json.load(f)

# Extract synsets and class labels (synsets are the 'wnid' and class labels are the first entry in 'synset')
synsets = [item['wnid'] for item in class_info]
class_labels = [item['synset'][0] for item in class_info]  # Using the first name in the 'synset' list

# Save the synsets and class labels to a txt file
synsets_path = r"E:\Thesis\IRTNet\data\Synsets\ImageNet_V2_synsets.txt"
with open(synsets_path, 'w') as f:
    for synset in synsets:
        f.write(f"{synset}\n")

print(f"Synset list saved to {synsets_path}")



# Path to the ImageNet R validation directory
imagenet_r_dir = r"E:\Thesis\IRTNet\data\ImageNetR\imagenet-r"  

# Get the list of synsets (folder names)
synsets = [folder for folder in os.listdir(imagenet_r_dir) if os.path.isdir(os.path.join(imagenet_r_dir, folder))]

# Save the synsets to a txt file
synsets_path = r"E:\Thesis\IRTNet\data\Synsets\ImageNet_R_synsets.txt"
with open(synsets_path, 'w') as f:
    for synset in synsets:
        f.write(f"{synset}\n")

print(f"Synset list saved to {synsets_path}")


# Paths
classes_file = r"E:\Thesis\IRTNet\data\ImageNot\imagenot_classes.txt"  # Path to the ImageNot classes.txt file
synsets_output = r"E:\Thesis\IRTNet\data\Synsets\ImageNot_synsets.txt"  # Desired path for the synsets file

def convert_classes_to_synsets(classes_path, output_path):
    # Read classes.txt
    with open(classes_path, 'r') as f:
        synsets = f.readlines()
    
    # Clean up and write to the new file
    with open(output_path, 'w') as f:
        for synset in synsets:
            synset_cleaned = synset.strip()  # Remove any extra whitespace
            f.write(f"{synset_cleaned}\n")
    
    print(f"Synsets saved to {output_path}")

# Run the conversion
convert_classes_to_synsets(classes_file, synsets_output)
