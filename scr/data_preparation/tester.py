# import scipy.io
# meta_file_path = r"E:\Thesis\IRTNet\data\ImageNet\ILSVRC\Annotations\CLS-LOC\val\ILSVRC2012_devkit_t12\data\meta.mat"
# meta = scipy.io.loadmat(meta_file_path)
# synsets = [item[0][0] for item in meta['synsets']['WNID']]
# # print(synsets[:10])  # Check the first 10 synsets

# # Output:
# # [np.str_('n02119789'), np.str_('n02100735'), np.str_('n02110185'), np.str_('n02096294'), np.str_('n02102040'), np.str_('n02066245'), np.str_('n02509815'), np.str_('n02124075'), np.str_('n02417914'), np.str_('n02123394')]

# ground_truth_file_path = r"E:\Thesis\IRTNet\data\ImageNet\ILSVRC\Annotations\CLS-LOC\val\ILSVRC2012_devkit_t12\data\ILSVRC2012_validation_ground_truth.txt"
# with open(ground_truth_file_path, 'r') as f:
#     ground_truth_indices = [int(line.strip()) for line in f]
# # print(ground_truth_indices[:10])

# # Output:
# # [490, 361, 231, 236, 141, 736, 491, 238, 239, 734]

# from PIL import Image
# import json

# # Load the synset-to-label mapping
# with open(r"E:/Thesis/IRTNet/data/ImageNet/imagenet_class_index.json", "r") as f:
#     imagenet_labels = json.load(f)

# synset = synsets[ground_truth_indices[0] - 1]  # First image synset
# class_name = next((label[1] for label in imagenet_labels.values() if label[0] == synset), None)
# # print(f"Synset: {synset}, Class name: {class_name}")

# # Output:
# # Synset: n01751748, Class name: sea_snake

# for i in range(10):  # First 10 ground truth indices
#     index = ground_truth_indices[i] - 1  # Convert to 0-based index
#     synset = synsets[index]
#     class_name = next((label[1] for label in imagenet_labels.values() if label[0] == synset), None)
#     print(f"Ground Truth Index: {ground_truth_indices[i]}, Synset: {synset}, Class Name: {class_name}")

# print(synsets[:100])  # Check the first 100 synsets

# import os
# from PIL import Image

# import os
# from PIL import Image

# import os
# from PIL import Image

# image_folder = r"E:\Thesis\IRTNet\data\ImageNet\ILSVRC\ILSVRC2012_img_val"
# ground_truth_file = r"E:\Thesis\IRTNet\data\ImageNet\ILSVRC\Annotations\CLS-LOC\val\ILSVRC2012_devkit_t12\data\ILSVRC2012_validation_ground_truth.txt"

# # Read ground truth indices
# with open(ground_truth_file, 'r') as f:
#     ground_truth_indices = [int(line.strip()) for line in f]

# # Get the first 10 images and map them to synsets
# for i, img_name in enumerate(sorted(os.listdir(image_folder))[:10]):
#     ground_truth_index = ground_truth_indices[i]
#     synset = synsets[ground_truth_index - 1]
#     print(f"Image: {img_name}, Ground Truth Index: {ground_truth_index}, Synset: {synset}")



import os
import shutil

# Paths
val_dir = "data/ImageNet/Reorganized_val"  # Directory containing validation images
ground_truth_file = "data/ImageNet/ILSVRC/Annotations/CLS-LOC/val/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt"  # File with numeric true labels
synset_mapping_file = "data/Synsets/ImageNet_synsets.txt"  # Maps indices to synsets

# Step 1: Load the ground truth labels
with open(ground_truth_file, "r") as f:
    ground_truth_labels = [int(line.strip()) for line in f]  # Numeric labels (1-based indexing)

# Step 2: Load the synset mapping
with open(synset_mapping_file, "r") as f:
    synset_mapping = {idx + 1: line.strip() for idx, line in enumerate(f)}  # 1-based indexing

# Step 3: Map numeric labels to synsets
true_synsets = [synset_mapping[label] for label in ground_truth_labels]

# Step 4: Reorganize images into synset folders
images = sorted(os.listdir(val_dir))  # Ensure images are sorted by name (e.g., ILSVRC2012_val_00000001.JPEG)
if len(images) != len(true_synsets):
    raise ValueError("Number of images does not match the number of labels!")

for image, synset in zip(images, true_synsets):
    synset_dir = os.path.join(val_dir, synset)
    if not os.path.exists(synset_dir):
        os.makedirs(synset_dir)  # Create directory for synset if it doesn't exist

    source = os.path.join(val_dir, image)
    destination = os.path.join(synset_dir, image)
    shutil.move(source, destination)  # Move image to the correct synset folder

print("Reorganization complete!")


# # Load the synset mapping
# synset_mapping = {}
# with open("data/Synsets/ImageNet_synsets.txt", "r") as f:
#     for idx, line in enumerate(f, start=1):  # Assuming 1-based indexing
#         synset = line.strip()
#         synset_mapping[idx] = synset

# # Example ground truth indices
# ground_truth_indices = [490, 361, 171, 822, 297, 482, 13, 704, 599, 164, 649]

# # Map indices to synsets
# ground_truth_synsets = [synset_mapping[idx] for idx in ground_truth_indices]
# print(ground_truth_synsets)

