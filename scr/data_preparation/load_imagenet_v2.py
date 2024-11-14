import os
from PIL import Image
import torch
from torchvision import transforms

def load_imagenet_v2_data(path):
    images, labels = [], []
    for class_name in os.listdir(path):
        class_path = os.path.join(path, class_name)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                image = Image.open(img_path)
                images.append(image)
                labels.append(class_name)
    return images, labels

def preprocess_imagenet_v2(image):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return preprocess(image)
