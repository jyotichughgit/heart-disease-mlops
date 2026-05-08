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

DATA_DIR  = "data"
DATA_FILE = "heart.csv"
DATA_PATH = os.path.join(DATA_DIR, DATA_FILE)
ZIP_URL   = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
ZIP_PATH  = os.path.join(DATA_DIR, "heart_disease.zip")

# Official UCI Cleveland 14 attribute names (no headers in raw file)
COLUMN_NAMES = [
    "age",      # age in years
    "sex",      # 1=male, 0=female
    "cp",       # chest pain type (0-3)
    "trestbps", # resting blood pressure (mmHg)
    "chol",     # serum cholesterol (mg/dl)
    "fbs",      # fasting blood sugar >120 mg/dl (1=true)
    "restecg",  # resting ECG results (0-2)
    "thalach",  # max heart rate achieved
    "exang",    # exercise induced angina (1=yes)
    "oldpeak",  # ST depression induced by exercise
    "slope",    # slope of peak exercise ST segment
    "ca",       # number of major vessels (0-3) — HAS MISSING VALUES (?)
    "thal",     # thalassemia — HAS MISSING VALUES (?)
    "target",   # diagnosis (0=no disease, 1-4=disease)
]

# Fallback URLs if UCI zip fails
FALLBACK_URLS = [
    "https://raw.githubusercontent.com/rashida048/Datasets/master/heart.csv",
    "https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/master/heart.csv",
]

MAX_RETRIES  = 3
RETRY_DELAY  = 5


def download_with_retry(url, dest):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(f"Attempt {attempt}/{MAX_RETRIES}: {url}")
            urllib.request.urlretrieve(url, dest)
            return True
        except Exception as e:
            logger.warning(f"Attempt {attempt} failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
    return False


# Feature type definitions for missing value treatment
NUMERICAL_COLS   = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values by feature type:
    - Numerical columns  → fill with median
    - Categorical columns → fill with mode (most frequent value)
    """
    # Numerical: fill with median
    for col in NUMERICAL_COLS:
        if col in df.columns and df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.info(f"  {col}: filled missing with median={median_val}")

    # Categorical: fill with mode
    for col in CATEGORICAL_COLS:
        if col in df.columns and df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            logger.info(f"  {col}: filled missing with mode={mode_val}")

    return df


def process_cleveland_raw(raw_path):
    """
    Read raw Cleveland file which uses '?' for missing values.
    Steps:
      1. Read file — '?' becomes NaN
      2. Log missing values before filling (for EDA awareness)
      3. Fill numerical missing with median
      4. Fill categorical missing with mode
      5. Binarize target (0-4 → 0/1)
      6. Save cleaned CSV
    """
    # Step 1 — Read raw file: '?' → NaN
    df = pd.read_csv(
        raw_path,
        header=None,
        names=COLUMN_NAMES,
        na_values="?",
        encoding="latin-1"
    )
    logger.info(f"Raw shape: {df.shape}")

    # Step 2 — Log missing values BEFORE filling
    missing_before = df.isnull().sum()
    missing_cols = missing_before[missing_before > 0]
    if len(missing_cols) > 0:
        logger.info("Missing values found (before filling):")
        for col, count in missing_cols.items():
            feature_type = "numerical" if col in NUMERICAL_COLS else "categorical"
            logger.info(f"  {col}: {count} missing ({feature_type})")
    else:
        logger.info("No missing values found in raw file")

    # Step 3 & 4 — Fill missing values by feature type
    logger.info("Filling missing values:")
    df = fill_missing_values(df)

    # Verify no missing values remain
    assert df.isnull().sum().sum() == 0, "Missing values still present after filling!"
    logger.info("All missing values filled successfully")

    # Step 5 — Binarize target: 0=no disease, 1=disease (original is 0-4)
    df["target"] = (df["target"] > 0).astype(int)

    # Step 6 — Save cleaned CSV
    df.to_csv(DATA_PATH, index=False)
    logger.info(f"Cleaned dataset saved to {DATA_PATH}")
    logger.info(f"Final shape: {df.shape}")
    logger.info(f"Target distribution: {df['target'].value_counts().to_dict()}")
    return True


def try_zip_source():
    """Try downloading from official UCI zip."""
    if not download_with_retry(ZIP_URL, ZIP_PATH):
        return False

    extract_dir = os.path.join(DATA_DIR, "extracted")
    try:
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            z.extractall(extract_dir)
            logger.info(f"Extracted: {z.namelist()}")

        # Find processed.cleveland.data exactly
        cleveland_path = None
        for root, _, files in os.walk(extract_dir):
            for f in files:
                if f == "processed.cleveland.data":
                    cleveland_path = os.path.join(root, f)
                    break

        if not cleveland_path:
            logger.warning("processed.cleveland.data not found in zip")
            return False

        logger.info(f"Found: {cleveland_path}")
        return process_cleveland_raw(cleveland_path)

    except Exception as e:
        logger.warning(f"Zip processing failed: {e}")
        return False
    finally:
        if os.path.exists(ZIP_PATH):
            os.remove(ZIP_PATH)
        if os.path.exists(extract_dir):
            shutil.rmtree(extract_dir)


def try_fallback(url):
    """Try downloading a pre-cleaned fallback CSV."""
    raw = os.path.join(DATA_DIR, "raw.csv")
    if not download_with_retry(url, raw):
        return False
    try:
        df = pd.read_csv(raw)
        if "target" not in df.columns:
            df = df.rename(columns={df.columns[-1]: "target"})
        df["target"] = (df["target"] > 0).astype(int)

        missing = df.isnull().sum()
        missing_cols = missing[missing > 0]
        if len(missing_cols) > 0:
            logger.info(f"Missing values found: {missing_cols.to_dict()}")
        else:
            logger.warning(
                "Fallback URL has no missing values — "
                "this is a pre-cleaned version. "
                "Missing values (ca, thal) were already filled upstream."
            )

        df.to_csv(DATA_PATH, index=False)
        logger.info(f"Saved from fallback: {DATA_PATH} | Shape: {df.shape}")
        return True
    except Exception as e:
        logger.warning(f"Fallback failed: {e}")
        return False
    finally:
        if os.path.exists(raw):
            os.remove(raw)


def download_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(DATA_PATH):
        logger.info(f"Dataset already exists at {DATA_PATH}")
        # Check if existing file has missing values
        df = pd.read_csv(DATA_PATH)
        missing = df.isnull().sum()
        missing_cols = missing[missing > 0]
        if len(missing_cols) > 0:
            logger.info(f"Existing file has missing values: {missing_cols.to_dict()}")
        else:
            logger.warning(
                "Existing heart.csv has NO missing values. "
                "It may have been downloaded from a pre-cleaned source. "
                "Delete data/heart.csv and re-run to get the raw UCI file."
            )
        return True

    logger.info("Trying primary source: UCI zip...")
    if try_zip_source():
        return True

    for url in FALLBACK_URLS:
        logger.info(f"Trying fallback: {url}")
        if try_fallback(url):
            return True

    logger.error(
        "All sources failed. Download manually from: "
        "https://archive.ics.uci.edu/dataset/45/heart+disease"
    )
    return False


if __name__ == "__main__":
    success = download_dataset()
    exit(0 if success else 1)
