"""
scripts/validate_model.py
Validates trained model meets performance threshold before deployment.
Usage: python scripts/validate_model.py --threshold 0.80
"""

import os
import sys
import json
import pickle
import argparse
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def validate(threshold: float = 0.80):
    model_path  = "src/models/best_model.pkl"
    config_path = "src/models/feature_config.json"
    data_path   = "data/heart.csv"

    # Load config
    with open(config_path) as f:
        config = json.load(f)

    logger.info(f"Model: {config['best_model']} | Reported ROC-AUC: {config['best_roc_auc']}")

    # Load model
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Load data and re-evaluate
    df = pd.read_csv(data_path)
    if "target" not in df.columns:
        df = df.rename(columns={df.columns[-1]: "target"})
    df["target"] = (df["target"] > 0).astype(int)

    all_feats = config["all_features"]
    X = df[all_feats]
    y = df["target"]

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    y_proba = model.predict_proba(X_test)[:, 1]
    auc     = roc_auc_score(y_test, y_proba)

    logger.info(f"Validation ROC-AUC: {auc:.4f} | Threshold: {threshold}")

    if auc >= threshold:
        logger.info(f"✅ Model PASSED validation (AUC={auc:.4f} >= {threshold})")
        return True
    else:
        logger.error(f"❌ Model FAILED validation (AUC={auc:.4f} < {threshold})")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.80)
    args = parser.parse_args()
    passed = validate(args.threshold)
    sys.exit(0 if passed else 1)
