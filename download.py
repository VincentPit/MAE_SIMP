"""
COCO Dataset Downloader

This script downloads the COCO 2017 unlabeled dataset (~20GB) for training
vision transformer models. It includes progress tracking, error handling,
and automatic extraction.

Usage:
    python download.py
"""

import os
import sys
import time
import zipfile
from pathlib import Path

try:
    import requests
except ImportError:
    print("Error: requests library not found. Install it with: pip install requests")
    sys.exit(1)

# Dataset configuration
COCO_URL = "http://images.cocodataset.org/zips/unlabeled2017.zip"
ZIP_FILENAME = "unlabeled2017.zip"
EXTRACT_DIR = "unlabeled2017"
EXPECTED_SIZE = 19_993_294_617  # Expected file size in bytes (~20GB)


def download_file_with_progress(url: str, filename: str) -> bool:
    """
    Download a file with progress tracking and error handling.
    
    Args:
        url: URL to download from
        filename: Local filename to save to
        
    Returns:
        bool: True if download successful, False otherwise
    """
    try:
        print(f"Starting download from: {url}")
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        if total_size > 0:
            print(f"Total size: {total_size / (1024**3):.2f} GB")
        
        downloaded_size = 0
        start_time = time.time()
        
        with open(filename, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    downloaded_size += len(chunk)
                    
                    # Progress update every 100MB
                    if downloaded_size % (100 * 1024 * 1024) == 0 or downloaded_size == total_size:
                        if total_size > 0:
                            percent = (downloaded_size / total_size) * 100
                            elapsed = time.time() - start_time
                            speed = downloaded_size / (1024 * 1024) / elapsed  # MB/s
                            print(f"Progress: {percent:.1f}% ({downloaded_size / (1024**3):.2f} GB) - Speed: {speed:.1f} MB/s")
        
        print("Download completed successfully!")
        return True
        
    except requests.RequestException as e:
        print(f"Download failed: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error during download: {e}")
        return False


def extract_zip(zip_path: str, extract_to: str) -> bool:
    """
    Extract ZIP file with progress tracking.
    
    Args:
        zip_path: Path to ZIP file
        extract_to: Directory to extract to
        
    Returns:
        bool: True if extraction successful, False otherwise
    """
    try:
        print("Starting extraction...")
        
        # Create extraction directory
        os.makedirs(extract_to, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            total_files = len(file_list)
            print(f"Extracting {total_files} files...")
            
            for i, file_info in enumerate(file_list):
                zip_ref.extract(file_info, extract_to)
                
                # Progress update every 1000 files
                if (i + 1) % 1000 == 0 or (i + 1) == total_files:
                    percent = ((i + 1) / total_files) * 100
                    print(f"Extraction progress: {percent:.1f}% ({i + 1}/{total_files} files)")
        
        print("Extraction completed successfully!")
        return True
        
    except zipfile.BadZipFile:
        print("Error: Downloaded file is not a valid ZIP archive")
        return False
    except Exception as e:
        print(f"Error during extraction: {e}")
        return False


def main():
    """
    Main function to download and extract COCO dataset.
    """
    print("=" * 50)
    print("COCO 2017 Unlabeled Dataset Downloader")
    print("=" * 50)
    
    # Check if dataset already exists
    if os.path.exists(EXTRACT_DIR) and os.listdir(EXTRACT_DIR):
        print(f"Dataset already exists in '{EXTRACT_DIR}'. Skipping download.")
        return
    
    # Check available disk space
    try:
        free_space = os.statvfs('.').f_bavail * os.statvfs('.').f_frsize
        required_space = EXPECTED_SIZE * 1.5  # ZIP + extracted files
        
        if free_space < required_space:
            print(f"Warning: Insufficient disk space. Required: {required_space / (1024**3):.1f} GB, Available: {free_space / (1024**3):.1f} GB")
            response = input("Continue anyway? (y/N): ")
            if response.lower() != 'y':
                print("Download cancelled.")
                return
    except (OSError, AttributeError):
        print("Could not check disk space. Proceeding with download...")
    
    # Download the dataset
    if not download_file_with_progress(COCO_URL, ZIP_FILENAME):
        print("Download failed. Please check your internet connection and try again.")
        return
    
    # Verify file size
    actual_size = os.path.getsize(ZIP_FILENAME)
    print(f"Downloaded file size: {actual_size / (1024**3):.2f} GB")
    
    # Extract the dataset
    if not extract_zip(ZIP_FILENAME, EXTRACT_DIR):
        print("Extraction failed. Please check the downloaded file and try again.")
        return
    
    # Clean up ZIP file
    try:
        os.remove(ZIP_FILENAME)
        print("ZIP file removed to save disk space.")
    except OSError as e:
        print(f"Warning: Could not remove ZIP file: {e}")
    
    # Final verification
    dataset_path = Path(EXTRACT_DIR) / "unlabeled2017"
    if dataset_path.exists():
        image_count = len(list(dataset_path.glob("*.jpg")))
        print(f"\nDataset ready! Found {image_count} images in '{dataset_path}'")
    else:
        print(f"\nWarning: Could not find expected directory structure in '{EXTRACT_DIR}'")
    
    print("\n" + "="*50)
    print("Download and extraction completed!")
    print("="*50)


if __name__ == "__main__":
    main()
