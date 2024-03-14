#!/bin/bash
zip_path="/home/bouhsi95/contrastive/data/raw/mnist.zip"
source_folder="/home/bouhsi95/contrastive/data/raw/mnist"
destination_folder="/home/bouhsi95/contrastive/data/processed/mnist"
# Set default values for source and destination folders
# unzip "$zip_path" -d "$source_folder"

# Run the Python script to copy folders
python /home/bouhsi95/contrastive/src/utils/preprocessing/preprocess_data.py "$source_folder" "$destination_folder"
