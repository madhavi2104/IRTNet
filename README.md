# IRTNet: Applying Item Response Theory to ImageNet Data Models

This repository is part of my master's thesis, focusing on using Item Response Theory (IRT) to analyze the robustness of machine learning models across various datasets derived from ImageNet. Due to dataset size constraints, this repository does not include the datasets directly but provides instructions on downloading and preparing them for use.

## **Repository Structure**

```
IRTNet/
├── README.md                        # Repository guide and instructions
├── requirements.txt                 # Python dependencies
│
├── data/                            # Dataset-related files
│   ├── ImageNet/                    # ImageNet validation dataset
│   ├── ImageNet_Sketch/             # ImageNet Sketch dataset
│   ├── ImageNet_V2/                 # ImageNet V2 dataset
│   ├── ImageNet_R/                  # ImageNet R dataset
│   ├── ImageNot/                    # ImageNot dataset
│   ├── Synsets/                     # Generated synset files for all datasets
│   │   ├── ImageNet_synsets.txt
│   │   ├── Sketch_synsets.txt
│   │   ├── ImageNet_V2_synsets.txt
│   │   ├── ImageNet_R_synsets.txt
│   │   ├── ImageNot_synsets.txt
│
├── src/                             # Source code for data processing and analysis
│   ├── synset_generator.py          # Synset extraction and generation scripts
│   ├── data_preparation/            # Scripts for dataset preprocessing
│   ├── model_analysis.py            # IRT and model evaluation scripts
│   ├── visualization.py             # Visualization utilities
│
├── notebooks/                       # Jupyter notebooks for experiments and demos
│
├── results/                         # Outputs such as plots and tables
│
└── logs/                            # Logs for debugging and monitoring

```


