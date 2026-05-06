"""
scripts/download_data.py
Downloads the Heart Disease dataset from the official UCI ML Repository.
Source: https://archive.ics.uci.edu/dataset/45/heart+disease
Usage: python scripts/download_data.py
"""

import logging
import os
import shutil
import time
import urllib.request
import zipfile

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = "data"
DATA_FILE = "heart.csv"
DATA_PATH = os.path.join(DATA_DIR, DATA_FILE)
ZIP_PATH = os.path.join(DATA_DIR, "heart_disease.zip")

# Primary: official UCI zip
# Fallbacks: direct Cleveland processed file mirrors
SOURCES = [
    {
        "type": "zip",
        "url": "https://archive.ics.uci.edu/static/public/45/heart+disease.zip",
        "target_file": "processed.cleveland.data",
    },
    {
        "type": "csv",
        "url": "https://raw.githubusercontent.com/tbrugere/automl_sr/master/datasets/heart-disease/processed.cleveland.data",
    },
    {
        "type": "csv",
        "url": "https://raw.githubusercontent.com/benjaminmgross/visualize-wealth/master/visualize_wealth/data/heart-disease/processed.cleveland.data",
    },
]

# Official 14 attribute names for processed Cleveland dataset
COLUMN_NAMES = [
    "age",       # age in years
    "sex",       # 1=male, 0=female
    "cp",        # chest pain type (0-3)
    "trestbps",  # resting blood pressure (mmHg)
    "chol",      # serum cholesterol (mg/dl)
    "fbs",       # fasting blood sugar >120 mg/dl (1=true)
    "restecg",   # resting ECG results (0-2)
    "thalach",   # max heart rate achieved
    "exang",     # exercise induced angina (1=yes)
    "oldpeak",   # ST depression induced by exercise
    "slope",     # slope of peak exercise ST segment
    "ca",        # number of major vessels coloured by fluoroscopy (0-3)
    "thal",      # thalassemia (3=normal, 6=fixed defect, 7=reversable defect)
    "target",    # diagnosis (0=no disease, 1-4=disease present)
]

MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds


def download_with_retry(url, dest_path):
    """Download a file with retries on failure."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(f"Attempt {attempt}/{MAX_RETRIES}: {url}")
            urllib.request.urlretrieve(url, dest_path)
            logger.info(f"Downloaded successfully to {dest_path}")
            return True
        except Exception as e:
            logger.warning(f"Attempt {attempt} failed: {e}")
            if attempt < MAX_RETRIES:
                logger.info(f"Retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
    return False


def process_cleveland_file(raw_path):
    """Read raw Cleveland data, clean and save as heart.csv."""
    df = pd.read_csv(raw_path, header=None, names=COLUMN_NAMES, na_values="?", encoding="latin-1")
    logger.info(f"Raw shape: {df.shape}")
    logger.info(f"Missing values:\n{df.isnull().sum()}")

    # Binarize target: 0=no disease, 1=disease (original is 0-4)
    df["target"] = (df["target"] > 0).astype(int)

    # Drop rows missing more than 2 values, fill rest with median
    df.dropna(thresh=len(COLUMN_NAMES) - 2, inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)

    df.to_csv(DATA_PATH, index=False)
    logger.info(f"Clean dataset saved to {DATA_PATH}")
    logger.info(f"Final shape: {df.shape} | Target: {df['target'].value_counts().to_dict()}")
    return True


def try_zip_source(source):
    """Try downloading and extracting the UCI zip file."""
    success = download_with_retry(source["url"], ZIP_PATH)
    if not success:
        return False

    extract_dir = os.path.join(DATA_DIR, "extracted")
    try:
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            z.extractall(extract_dir)
            logger.info(f"Extracted files: {z.namelist()}")

        # Find processed.cleveland.data exactly
        cleveland_path = None
        for root, dirs, files in os.walk(extract_dir):
            for f in files:
                if f == source["target_file"]:
                    cleveland_path = os.path.join(root, f)
                    break

        if not cleveland_path:
            logger.warning(f"{source['target_file']} not found in zip")
            return False

        logger.info(f"Found: {cleveland_path}")
        result = process_cleveland_file(cleveland_path)
        return result

    except Exception as e:
        logger.warning(f"Zip processing failed: {e}")
        return False
    finally:
        if os.path.exists(ZIP_PATH):
            os.remove(ZIP_PATH)
        if os.path.exists(extract_dir):
            shutil.rmtree(extract_dir)


def try_csv_source(source):
    """Try downloading a direct .data file."""
    raw_path = os.path.join(DATA_DIR, "raw.data")
    success = download_with_retry(source["url"], raw_path)
    if not success:
        return False
    try:
        result = process_cleveland_file(raw_path)
        return result
    except Exception as e:
        logger.warning(f"CSV processing failed: {e}")
        return False
    finally:
        if os.path.exists(raw_path):
            os.remove(raw_path)


def download_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(DATA_PATH):
        logger.info(f"Dataset already exists at {DATA_PATH}")
        return True

    for source in SOURCES:
        logger.info(f"Trying source: {source['url']}")
        if source["type"] == "zip":
            success = try_zip_source(source)
        else:
            success = try_csv_source(source)

        if success:
            logger.info("Dataset downloaded and processed successfully!")
            return True
        logger.warning(f"Source failed, trying next...")

    logger.error(
        "All sources failed. Download manually from: "
        "https://archive.ics.uci.edu/dataset/45/heart+disease"
    )
    return False


if __name__ == "__main__":
    success = download_dataset()
    exit(0 if success else 1)
