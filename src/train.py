"""
train.py - Standalone training script for CI/CD pipeline.
Usage: python src/train.py --data data/heart.csv --output src/models/
"""

import argparse
import json
import logging
import os
import pickle
import warnings

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

NUMERICAL_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
TARGET = "target"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data(path: str) -> pd.DataFrame:
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path)
    if TARGET not in df.columns:
        df = df.rename(columns={df.columns[-1]: TARGET})
    df[TARGET] = (df[TARGET] > 0).astype(int)
    logger.info(f"Shape: {df.shape} | Target: {df[TARGET].value_counts().to_dict()}")
    return df


def build_preprocessor(num_feats, cat_feats):
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_feats,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                cat_feats,
            ),
        ]
    )


def train_and_evaluate(pipeline, X_train, y_train, X_test, y_test, cv):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "cv_roc_auc_mean": cv_scores.mean(),
        "cv_roc_auc_std": cv_scores.std(),
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


def save_confusion_matrix(y_test, y_pred, name, plots_dir):
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, ax=ax, display_labels=["No Disease", "Disease"], colorbar=False, cmap="Blues"
    )
    ax.set_title(f"{name} - Confusion Matrix", fontweight="bold")
    plt.tight_layout()
    path = os.path.join(plots_dir, f"{name}_confusion_matrix.png")
    plt.savefig(path, dpi=120)
    plt.close()
    mlflow.log_artifact(path, artifact_path="plots")


def save_roc_curve(y_test, y_proba, name, plots_dir):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#c0392b", lw=2.5, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{name} - ROC Curve", fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(plots_dir, f"{name}_roc_curve.png")
    plt.savefig(path, dpi=120)
    plt.close()
    mlflow.log_artifact(path, artifact_path="plots")


def save_feature_importance(pipeline, name, num_feats, cat_feats, plots_dir):
    clf = pipeline.named_steps["classifier"]
    if not hasattr(clf, "feature_importances_"):
        return
    try:
        ohe = pipeline.named_steps["preprocessor"].named_transformers_["cat"]["encoder"]
        feat_names = num_feats + ohe.get_feature_names_out(cat_feats).tolist()
        importances = pd.Series(clf.feature_importances_, index=feat_names).sort_values(ascending=False)[:15]
        fig, ax = plt.subplots(figsize=(10, 6))
        importances.plot(kind="barh", ax=ax, color="#4C72B0", edgecolor="white")
        ax.set_title(f"{name} - Top 15 Feature Importances", fontweight="bold")
        ax.invert_yaxis()
        plt.tight_layout()
        path = os.path.join(plots_dir, f"{name}_feature_importance.png")
        plt.savefig(path, dpi=120)
        plt.close()
        mlflow.log_artifact(path, artifact_path="plots")
    except Exception as e:
        logger.warning(f"Feature importance plot failed: {e}")


def main(data_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment("Heart-Disease-Classification")

    df = load_data(data_path)
    num_feats = [f for f in NUMERICAL_FEATURES if f in df.columns]
    cat_feats = [f for f in CATEGORICAL_FEATURES if f in df.columns]
    all_feats = num_feats + cat_feats

    X = df[all_feats]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    preprocessor = build_preprocessor(num_feats, cat_feats)

    models = {
        "logistic_regression": Pipeline(
            [
                ("preprocessor", preprocessor),
                ("classifier", LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocessor", preprocessor),
                ("classifier", RandomForestClassifier(n_estimators=200, max_depth=8, random_state=RANDOM_STATE)),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    GradientBoostingClassifier(
                        n_estimators=150, learning_rate=0.1, max_depth=4, random_state=RANDOM_STATE
                    ),
                ),
            ]
        ),
    }

    results = {}
    for name, pipeline in models.items():
        logger.info(f"Training {name}...")
        with mlflow.start_run(run_name=name):
            # Log parameters
            mlflow.log_params(
                {
                    "model_type": name,
                    "test_size": TEST_SIZE,
                    "random_state": RANDOM_STATE,
                    "cv_folds": 5,
                    "numerical_features": str(num_feats),
                    "categorical_features": str(cat_feats),
                }
            )
            # Train and evaluate
            metrics = train_and_evaluate(pipeline, X_train, y_train, X_test, y_test, cv)
            # Log metrics
            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, float)})
            # Log plots as artifacts
            save_confusion_matrix(y_test, metrics["y_pred"], name, plots_dir)
            save_roc_curve(y_test, metrics["y_proba"], name, plots_dir)
            save_feature_importance(pipeline, name, num_feats, cat_feats, plots_dir)
            # Log model
            mlflow.sklearn.log_model(pipeline, name)
            results[name] = (pipeline, metrics["roc_auc"])
            logger.info(f"  ROC-AUC={metrics['roc_auc']:.4f} | CV-AUC={metrics['cv_roc_auc_mean']:.4f}")

    # Save best model
    best_name, (best_model, best_auc) = max(results.items(), key=lambda x: x[1][1])
    logger.info(f"Best model: {best_name} (ROC-AUC={best_auc:.4f})")

    with open(os.path.join(output_dir, "best_model.pkl"), "wb") as f:
        pickle.dump(best_model, f)

    config = {
        "best_model": best_name,
        "best_roc_auc": round(best_auc, 4),
        "numerical_features": num_feats,
        "categorical_features": cat_feats,
        "all_features": all_feats,
        "target": TARGET,
    }
    with open(os.path.join(output_dir, "feature_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"Model saved to {output_dir}")
    logger.info("View MLflow UI: mlflow ui --backend-store-uri file:./mlruns --port 5000")
    return best_auc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/heart.csv")
    parser.add_argument("--output", default="src/models/")
    args = parser.parse_args()
    main(args.data, args.output)
