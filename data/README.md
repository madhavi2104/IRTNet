# Data

## Datasets
The datasets used in this project include:
1. **ImageNet**
2. **ImageNet Sketch**
3. **ImageNet R**
4. **ImageNet V2**
5. **ImageNot**

Follow the instructions below to download and prepare these datasets.

---

### **1. ImageNet**
- **Location**: [`ILSVRC2012`](https://www.image-net.org/)
- **Files Needed**:
  - `ILSVRC2012_img_val/` (validation images)
  - `Annotations/CLS-LOC/val/ILSVRC2012_devkit_t12/` (annotations and `meta.mat` file)
- **Instructions**:
  1. Download the validation images and devkit from the official [ImageNet website](https://www.image-net.org/download.php).
  2. Organize the files as:
     ```
     data/
     └── ImageNet/
         ├── ILSVRC2012_img_val/
         └── Annotations/CLS-LOC/val/ILSVRC2012_devkit_t12/
     ```

---

### **2. ImageNet Sketch**
- **Location**: [ImageNet Sketch GitHub](https://github.com/HaohanWang/ImageNet-Sketch)
- **Instructions**:
  1. Clone the repository:
     ```bash
     git clone https://github.com/HaohanWang/ImageNet-Sketch.git
     ```
  2. Download the dataset using the provided script:
     ```bash
     python data_download.py
     ```
  3. Extract the dataset to:
     ```
     data/
     └── ImageNet_Sketch/
         ├── data/
         └── Synsets/
     ```

---

### **3. ImageNet R**
- **Location**: [ImageNet R GitHub](https://github.com/hendrycks/imagenet-r)
- **Instructions**:
  1. Download the dataset from the [GitHub releases page](https://github.com/hendrycks/imagenet-r/releases).
  2. Extract the images and organize as:
     ```
     data/
     └── ImageNet_R/
         ├── nXXXXXX/
         └── Synsets/
     ```

---

### **4. ImageNet V2**
- **Location**: [Hugging Face Dataset](https://huggingface.co/datasets/vaishaal/ImageNetV2)
- **Instructions**:
  1. Use the Hugging Face Hub to download:
     ```bash
     pip install datasets
     python -c "from datasets import load_dataset; load_dataset('vaishaal/ImageNetV2')"
     ```
  2. Save the dataset as:
     ```
     data/
     └── ImageNet_V2/
         ├── matched-frequency/
         ├── threshold-0.7/
         ├── top-images/
         └── Synsets/
     ```

---

### **5. ImageNot**
- **Location**: [ImageNot GitHub](https://github.com/olawalesalaudeen/imagenot)
- **Instructions**:
  1. Clone the repository:
     ```bash
     git clone https://github.com/olawalesalaudeen/imagenot.git
     ```
  2. Manually download the dataset following the repository's README.
  3. Save the dataset as:
     ```
     data/
     └── ImageNot/
         ├── images/
         └── Synsets/
     ```
*Currently, no dataset is available for ImageNot*

---

## **Synset Files**

Each dataset requires a **synset mapping file**. These files can be found in `data/Synsets/`:
- `ImageNet_synsets.txt`
- `Sketch_synsets.txt`
- `ImageNetR_synsets.txt`
- `ImageNetV2_synsets.txt`
- `ImageNot_synsets.txt`

If any are missing, you can regenerate them using the `synset_generato.py` file in the repository.

---

## **Setup**

1. Clone this repository:
   ```bash
   git clone https://github.com/madhavi2104/IRTNet.git
   cd IRTNet

