from PIL import Image
import os
from tqdm import tqdm

def resize_images(dataset_dir, output_dir, size=(224, 224)):
    """
    Resizes images to the specified size.

    Parameters:
    - dataset_dir: Path to the dataset directory.
    - output_dir: Path to save resized images.
    - size: Tuple specifying the new image dimensions (width, height).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, _, files in os.walk(dataset_dir):
        for file in tqdm(files, desc="Resizing images"):
            file_path = os.path.join(root, file)
            if not file.lower().endswith(('.jpeg', '.jpg', '.png')):
                continue

            try:
                img = Image.open(file_path)
                img_resized = img.resize(size)
                output_path = os.path.join(output_dir, os.path.relpath(file_path, dataset_dir))
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                img_resized.save(output_path)
            except Exception as e:
                print(f"Error resizing {file_path}: {e}")

if __name__ == "__main__":
    from config import DATASETS, IMAGE_RESIZE_DIM

    for dataset, path in DATASETS.items():
        output_dir = os.path.join(path, "Resized")
        resize_images(path, output_dir, size=IMAGE_RESIZE_DIM)
