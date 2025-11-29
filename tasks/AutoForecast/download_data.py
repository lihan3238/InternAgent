#!/usr/bin/env python3
"""
Data download script for AutoForecast task
Downloads the UCI Individual household electric power consumption dataset
"""

import os
import urllib.request
import zipfile
import shutil
from pathlib import Path

def download_and_extract_data():
    """Download and extract the energy consumption dataset"""

    # URLs and paths
    data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip"
    zip_path = "datasets/household_power_consumption.zip"
    extract_path = "datasets/"
    data_file = "datasets/household_power_consumption.txt"

    # Create datasets directory if it doesn't exist
    Path("datasets").mkdir(exist_ok=True)

    # Check if data already exists
    if os.path.exists(data_file):
        print(f"Data file already exists at {data_file}")
        return data_file

    print("Downloading energy consumption dataset...")
    print(f"From: {data_url}")

    try:
        # Download the zip file
        with urllib.request.urlopen(data_url) as response:
            with open(zip_path, 'wb') as f:
                shutil.copyfileobj(response, f)

        print(f"Downloaded to {zip_path}")

        # Extract the zip file
        print("Extracting data...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

        # Find the extracted txt file
        extracted_files = []
        for root, dirs, files in os.walk(extract_path):
            for file in files:
                if file.endswith('.txt'):
                    extracted_files.append(os.path.join(root, file))

        if extracted_files:
            txt_file = extracted_files[0]
            # Move to the expected location
            if txt_file != data_file:
                shutil.move(txt_file, data_file)
            print(f"Data extracted to {data_file}")
        else:
            raise FileNotFoundError("Could not find the extracted txt file")

        # Clean up
        if os.path.exists(zip_path):
            os.remove(zip_path)

        # Clean up any extra directories
        for item in os.listdir(extract_path):
            item_path = os.path.join(extract_path, item)
            if os.path.isdir(item_path) and item != "VOCdevkit":
                shutil.rmtree(item_path)

        return data_file

    except Exception as e:
        print(f"Error downloading data: {e}")
        print("\nPlease download the dataset manually:")
        print("1. Go to: https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption")
        print("2. Download the zip file")
        print("3. Extract household_power_consumption.txt to datasets/ directory")
        raise

if __name__ == "__main__":
    data_path = download_and_extract_data()
    print(f"\nDataset ready at: {data_path}")
    print("You can now run the training script!")
