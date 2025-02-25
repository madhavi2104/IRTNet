import scipy.io
import os
import json

# Path to meta.mat
meta_file = r"E:\Thesis\IRTNet\data\ImageNet\ILSVRC\ILSVRC2012_devkit_t12\data\meta.mat"

# Load meta.mat
meta = scipy.io.loadmat(meta_file)

# Extract synsets with ILSVRC2012_ID <= 1000
synsets = [entry[0][0] for entry in meta['synsets']['WNID'][:1000]]

# Save synsets to a text file
synset_file = r"E:\Thesis\IRTNet\data\Synsets\ImageNet_synsets.txt"
with open(synset_file, "w") as f:
    f.writelines(f"{synset}\n" for synset in synsets)

print(f"Synsets file created successfully with {len(synsets)} synsets!")


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

def extract_imagenet_synsets(meta_file, ground_truth_file, class_index_file, output_txt):
    """
    Extracts synsets and class names for the ImageNet validation dataset.

    Parameters:
    - meta_file: Path to meta.mat containing synset-class index mapping.
    - ground_truth_file: Path to ILSVRC2012_validation_ground_truth.txt.
    - class_index_file: Path to imagenet_class_index.json for synset-to-class-name mapping.
    - output_txt: Path to save the extracted synsets and class names.
    """
    # Load meta.mat to get the mapping of synsets to class indices
    meta = scipy.io.loadmat(meta_file)
    synsets = [item[0][0] for item in meta['synsets']['WNID']]

    # Load ground truth file
    with open(ground_truth_file, 'r') as f:
        ground_truth_indices = [int(line.strip()) for line in f]

    # Map ground truth indices to synsets
    ground_truth_synsets = [synsets[index - 1] for index in ground_truth_indices]

    # Load class index file to map synsets to class names
    with open(class_index_file, 'r') as f:
        imagenet_labels = json.load(f)

    # Create a mapping from synset to class name
    synset_to_class_name = {v[0]: v[1] for v in imagenet_labels.values()}

    # Save the unique synsets and their class names to a text file
    with open(output_txt, 'w') as f:
        f.write("Synset\tClass Name\n")  # Header row
        for synset in sorted(set(ground_truth_synsets)):
            class_name = synset_to_class_name.get(synset, "Unknown")
            f.write(f"{synset}\t{class_name}\n")

    print(f"Synsets and class names saved to {output_txt}")

# Example usage:
meta_file_path = r"E:\Thesis\IRTNet\data\ImageNet\ILSVRC\Annotations\CLS-LOC\val\ILSVRC2012_devkit_t12\data\meta.mat"
ground_truth_file_path = r"E:\Thesis\IRTNet\data\ImageNet\ILSVRC\Annotations\CLS-LOC\val\ILSVRC2012_devkit_t12\data\ILSVRC2012_validation_ground_truth.txt"
class_index_file_path = r"E:\Thesis\IRTNet\data\ImageNet\imagenet_class_index.json"
output_file_path = r"E:\Thesis\IRTNet\data\Synsets\ImageNet_synsets_and_classes.txt"

extract_imagenet_synsets(meta_file_path, ground_truth_file_path, class_index_file_path, output_file_path)

import os
import json

def extract_synsets_from_readme(readme_path, output_txt):
    """
    Extracts synsets from the ImageNet-A README file.
    
    Parameters:
    - readme_path: Path to the README file containing synset information.
    - output_txt: Path to save the extracted synsets list.
    """
    synsets = []
    
    with open(readme_path, "r") as f:
        lines = f.readlines()
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) > 1 and parts[0].startswith("n"):  # Synsets are WordNet IDs
            synsets.append(parts[0])
    
    # Save synsets to a text file
    with open(output_txt, "w") as f:
        for synset in synsets:
            f.write(f"{synset}\n")
    
    print(f"Synsets file created successfully with {len(synsets)} synsets!")

if __name__ == "__main__":
    readme_path = "data/imagenet-a/README.txt"  # Path to the README file
    output_path = "data/Synsets/ImageNet_A_synsets.txt"  # Output path for synset list
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Ensure directory exists
    extract_synsets_from_readme(readme_path, output_path)