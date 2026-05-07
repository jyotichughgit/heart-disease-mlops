"""
scripts/inference.py
Demonstrates loading the saved model and running predictions.
Proves full reproducibility of the saved pickle model.
Usage: python scripts/inference.py
"""

import json
import logging
import os
import pickle

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = "src/models/best_model.pkl"
CONFIG_PATH = "src/models/feature_config.json"
DATA_PATH = "data/heart.csv"


def load_model():
    """Load saved pickle model."""
    logger.info(f"Loading model from {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    logger.info("Model loaded successfully")
    return model


def load_config():
    """Load feature configuration."""
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    logger.info(f"Best model: {config['best_model']} | ROC-AUC: {config['best_roc_auc']}")
    return config


def predict_single(model, config, patient_data: dict) -> dict:
    """
    Run prediction on a single patient.
    Args:
        model:        loaded pickle model (sklearn Pipeline)
        config:       feature configuration dict
        patient_data: dict with patient feature values
    Returns:
        dict with prediction, label and confidence
    """
    all_features = config["all_features"]
    input_df = pd.DataFrame([{k: patient_data.get(k, 0) for k in all_features}])

    prediction = int(model.predict(input_df)[0])
    probas = model.predict_proba(input_df)[0]
    prob_disease = float(probas[1])
    prob_no_disease = float(probas[0])
    confidence = max(prob_disease, prob_no_disease)
    label = "Heart Disease" if prediction == 1 else "No Heart Disease"

    return {
        "prediction": prediction,
        "prediction_label": label,
        "confidence": round(confidence, 4),
        "probability_disease": round(prob_disease, 4),
        "probability_no_disease": round(prob_no_disease, 4),
    }


def predict_batch(model, config, df: pd.DataFrame) -> pd.DataFrame:
    """
    Run predictions on a batch of patients.
    Args:
        model:  loaded pickle model
        config: feature configuration dict
        df:     DataFrame with patient features
    Returns:
        DataFrame with predictions appended
    """
    all_features = config["all_features"]
    X = df[all_features]
    predictions = model.predict(X)
    probas = model.predict_proba(X)[:, 1]

    results = df.copy()
    results["prediction"] = predictions
    results["probability_disease"] = probas.round(4)
    results["prediction_label"] = results["prediction"].map({0: "No Heart Disease", 1: "Heart Disease"})
    return results


def reproducibility_check(model, config):
    """
    Proves model is reproducible by running same input
    multiple times and verifying identical output.
    """
    logger.info("Running reproducibility check...")

    sample_patient = {
        "age": 63.0, "sex": 1.0, "cp": 3.0, "trestbps": 145.0,
        "chol": 233.0, "fbs": 1.0, "restecg": 0.0, "thalach": 150.0,
        "exang": 0.0, "oldpeak": 2.3, "slope": 0.0, "ca": 0.0, "thal": 1.0,
    }

    # Run same prediction 5 times
    results = [predict_single(model, config, sample_patient) for _ in range(5)]

    # Check all predictions are identical
    predictions = [r["prediction"] for r in results]
    confidences = [r["confidence"] for r in results]

    all_same = len(set(predictions)) == 1 and len(set(confidences)) == 1

    if all_same:
        logger.info(f"Reproducibility check PASSED — same result across 5 runs")
        logger.info(f"  Prediction:  {results[0]['prediction_label']}")
        logger.info(f"  Confidence:  {results[0]['confidence']}")
    else:
        logger.error("Reproducibility check FAILED — results differ across runs!")

    return all_same


def main():
    # Check model files exist
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model not found at {MODEL_PATH}")
        logger.error("Run: python src/train.py --data data/heart.csv --output src/models/")
        return

    # Load model and config
    model = load_model()
    config = load_config()

    # ── Single patient prediction ─────────────────────────────────────────────
    logger.info("\n=== Single Patient Prediction ===")
    sample_patient = {
        "age": 63.0, "sex": 1.0, "cp": 3.0, "trestbps": 145.0,
        "chol": 233.0, "fbs": 1.0, "restecg": 0.0, "thalach": 150.0,
        "exang": 0.0, "oldpeak": 2.3, "slope": 0.0, "ca": 0.0, "thal": 1.0,
    }
    result = predict_single(model, config, sample_patient)
    for k, v in result.items():
        logger.info(f"  {k}: {v}")

    # ── Batch prediction ──────────────────────────────────────────────────────
    logger.info("\n=== Batch Prediction (first 5 rows of dataset) ===")
    df = pd.read_csv(DATA_PATH)
    if "target" not in df.columns:
        df = df.rename(columns={df.columns[-1]: "target"})
    df["target"] = (df["target"] > 0).astype(int)

    batch_results = predict_batch(model, config, df.head(5))
    print(batch_results[["age", "sex", "chol", "prediction_label", "probability_disease"]].to_string(index=False))

    # ── Reproducibility check ─────────────────────────────────────────────────
    logger.info("\n=== Reproducibility Check ===")
    passed = reproducibility_check(model, config)

    # ── Pipeline info ─────────────────────────────────────────────────────────
    logger.info("\n=== Pipeline Steps ===")
    for step_name, step_obj in model.steps:
        logger.info(f"  {step_name}: {type(step_obj).__name__}")

    logger.info("\nInference script complete!")
    logger.info(f"Reproducibility: {'PASSED' if passed else 'FAILED'}")


if __name__ == "__main__":
    main()
