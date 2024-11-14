# src/data_preparation/load_imagenet.py
import torchvision.datasets as datasets

def load_imagenet(data_dir):
    return datasets.ImageNet(data_dir, split='train')
