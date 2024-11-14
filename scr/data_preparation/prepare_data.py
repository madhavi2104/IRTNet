from load_imagenet import load_imagenet_data, preprocess_imagenet
from load_imagenot import load_imagenot_data, preprocess_imagenot
from load_imagenet_v2 import load_imagenet_v2_data, preprocess_imagenet_v2
from load_imagenet_r import load_imagenet_r_data, preprocess_imagenet_r
from load_imagenet_sketch import load_imagenet_sketch_data, preprocess_imagenet_sketch
from data_utils import save_processed_data

def load_all_datasets():
    datasets = []
    # Load and preprocess each dataset
    datasets.append(("ImageNet", load_imagenet_data("./data/imagenet")))
    datasets.append(("ImageNot", load_imagenot_data("./data/imagenot")))
    datasets.append(("ImageNet V2", load_imagenet_v2_data("./data/imagenet_v2")))
    datasets.append(("ImageNet-R", load_imagenet_r_data("./data/imagenet_r")))
    datasets.append(("ImageNet Sketch", load_imagenet_sketch_data("./data/imagenet_sketch")))
    return datasets

def generate_irt_ready_data(datasets, output_path):
    all_images, all_labels = [], []
    for name, (images, labels) in datasets:
        all_images.extend(images)
        all_labels.extend(labels)
    save_processed_data(all_images, all_labels, output_path)
