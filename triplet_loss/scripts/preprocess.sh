#!/bin/bash

# Set default values for source and destination folders
unzip /content/rice-image-dataset.zip -d ./dataset
source_folder="./dataset"
destination_folder="/content/mlascent_contrastive_learning/data/processed"

# Run the Python script to copy folders
python /content/mlascent_contrastive_learning/src/utils/preprocessing/preprocess_data.py "$source_folder" "$destination_folder"
