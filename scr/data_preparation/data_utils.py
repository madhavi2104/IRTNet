import torch
from sklearn.model_selection import train_test_split

def resize_image(image, size):
    return image.resize(size)

def normalize_image(image, mean, std):
    return transforms.Normalize(mean=mean, std=std)(image)

def split_data(images, labels, test_size=0.2):
    return train_test_split(images, labels, test_size=test_size)

def save_processed_data(images, labels, path):
    torch.save({"images": images, "labels": labels}, path)
