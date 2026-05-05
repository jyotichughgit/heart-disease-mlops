"""
scripts/download_data.py
Downloads the Heart Disease dataset from the official UCI ML Repository.
Source: https://archive.ics.uci.edu/dataset/45/heart+disease
Usage: python scripts/download_data.py
"""

import os
import zipfile
import urllib.request
import logging
import shutil
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR   = "data"
DATA_FILE  = "heart.csv"
DATA_PATH  = os.path.join(DATA_DIR, DATA_FILE)
ZIP_URL    = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
ZIP_PATH   = os.path.join(DATA_DIR, "heart_disease.zip")

# UCI Cleveland dataset has no header row — these are the official 14 attribute names
COLUMN_NAMES = [
    "age",      # age in years
    "sex",      # 1=male, 0=female
    "cp",       # chest pain type (0-3)
    "trestbps", # resting blood pressure (mmHg)
    "chol",     # serum cholesterol (mg/dl)
    "fbs",      # fasting blood sugar > 120 mg/dl (1=true)
    "restecg",  # resting ECG results (0-2)
    "thalach",  # max heart rate achieved
    "exang",    # exercise induced angina (1=yes)
    "oldpeak",  # ST depression induced by exercise
    "slope",    # slope of peak exercise ST segment
    "ca",       # number of major vessels coloured by fluoroscopy (0-3)
    "thal",     # thalassemia (3=normal, 6=fixed, 7=reversable)
    "target"    # diagnosis (0=no disease, 1-4=disease present)
]


def download_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(DATA_PATH):
        logger.info(f"Dataset already exists at {DATA_PATH}")
        return True

    # Step 1 — Download the zip file
    logger.info(f"Downloading from: {ZIP_URL}")
    urllib.request.urlretrieve(ZIP_URL, ZIP_PATH)
    logger.info(f"Zip downloaded to {ZIP_PATH}")

    # Step 2 — Extract zip
    extract_dir = os.path.join(DATA_DIR, "extracted")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(extract_dir)
        extracted_files = z.namelist()
        logger.info(f"Files in zip: {extracted_files}")

    # Step 3 — Locate EXACTLY processed.cleveland.data (not cleveland.data)
    cleveland_path = None
    for root, dirs, files in os.walk(extract_dir):
        for f in files:
            if f == "processed.cleveland.data":   # exact filename match
                cleveland_path = os.path.join(root, f)
                break

    if not cleveland_path:
        logger.error("Could not find processed.cleveland.data in the zip.")
        return False

    logger.info(f"Found: {cleveland_path}")

    # Step 4 — Read raw file (no headers, '?' for missing values, latin-1 encoding)
    df = pd.read_csv(
        cleveland_path,
        header=None,
        names=COLUMN_NAMES,
        na_values="?",
        encoding="latin-1"   # UCI file is not UTF-8
    )
    logger.info(f"Raw shape: {df.shape}")
    logger.info(f"Missing values:\n{df.isnull().sum()}")

    # Step 5 — Binarize target: 0=no disease, 1=disease (original values are 0-4)
    df["target"] = (df["target"] > 0).astype(int)

    # Step 6 — Handle missing values
    df.dropna(thresh=len(COLUMN_NAMES) - 2, inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Step 7 — Save as clean CSV
    df.to_csv(DATA_PATH, index=False)
    logger.info(f"Clean dataset saved to {DATA_PATH}")
    logger.info(f"Final shape: {df.shape} | Target distribution: {df['target'].value_counts().to_dict()}")

    # Step 8 — Cleanup
    os.remove(ZIP_PATH)
    shutil.rmtree(extract_dir)
    logger.info("Cleaned up temporary files.")

    return True


if __name__ == "__main__":
    success = download_dataset()
    exit(0 if success else 1)
