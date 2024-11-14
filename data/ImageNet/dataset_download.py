import os
import zipfile
import subprocess

os.environ['KAGGLE_CONFIG_DIR'] = os.path.join("data", "config")
data_dir = "data/imagenet"
os.makedirs(data_dir, exist_ok=True)

# Full path to kaggle command
kaggle_command = r"C:\path\to\kaggle.exe"  # replace with your actual path

try:
    subprocess.run([
        kaggle_command, "competitions", "download", 
        "-c", "imagenet-object-localization-challenge", 
        "-p", data_dir
    ], check=True)
except subprocess.CalledProcessError as e:
    print("Error downloading dataset:", e)

# Extract all zip files in the data directory
zip_files = [f for f in os.listdir(data_dir) if f.endswith('.zip')]

for file in zip_files:
    file_path = os.path.join(data_dir, file)
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    os.remove(file_path)  # Remove the zip file after extraction

