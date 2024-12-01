import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', '..', 'data')

DATASETS = {
    'ImageNet': os.path.join(DATA_DIR, 'ImageNet'),
    'ImageNet_R': os.path.join(DATA_DIR, 'ImageNetR'),
    'ImageNet_Sketch': os.path.join(DATA_DIR, 'ImageNet_Sketch'),
    'ImageNot': os.path.join(DATA_DIR, 'ImageNot'),
    'ImageNet_V2': os.path.join(DATA_DIR, 'ImageNet_V2'),
}

SYNSETS_DIR = os.path.join(DATA_DIR, 'Synsets')

# Verify paths
print(f"BASE_DIR: {BASE_DIR}")
print(f"DATA_DIR: {DATA_DIR}")
for dataset, path in DATASETS.items():
    print(f"{dataset}: {path} (Exists: {os.path.exists(path)})")
print(f"SYNSETS_DIR: {SYNSETS_DIR} (Exists: {os.path.exists(SYNSETS_DIR)})")

