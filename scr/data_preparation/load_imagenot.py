import os
from PIL import Image
import torch
from torchvision import transforms

def load_imagenot_data(path):
    images, labels = [], []
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        image = Image.open(img_path)
        images.append(image)
        labels.append("unknown")  # Assuming no specific labels in ImageNot
    return images, labels

def preprocess_imagenot(image):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return preprocess(image)
