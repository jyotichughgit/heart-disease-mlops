"""
scripts/download_data.py
Downloads the Heart Disease UCI dataset.
Usage: python scripts/download_data.py
"""

import os
import urllib.request
import hashlib
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR  = "data"
DATA_FILE = "heart.csv"
DATA_PATH = os.path.join(DATA_DIR, DATA_FILE)

# Primary and fallback URLs
URLS = [
    "https://raw.githubusercontent.com/dsrscientist/dataset1/master/heart_disease.csv",
    "https://raw.githubusercontent.com/Garima13a/Heart-Disease-Prediction/master/heart.csv",
]

def download_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(DATA_PATH):
        logger.info(f"Dataset already exists at {DATA_PATH}")
        return True

    for url in URLS:
        try:
            logger.info(f"Trying: {url}")
            urllib.request.urlretrieve(url, DATA_PATH)
            logger.info(f"Dataset downloaded to {DATA_PATH}")
            return True
        except Exception as e:
            logger.warning(f"Failed to download from {url}: {e}")

    logger.error("All download sources failed. Please download manually from UCI ML Repository.")
    return False

if __name__ == "__main__":
    success = download_dataset()
    exit(0 if success else 1)
