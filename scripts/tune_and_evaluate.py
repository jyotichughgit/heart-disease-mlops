"""
scripts/tune_and_evaluate.py
Hyperparameter tuning and model comparison for Task 2.
Saves comparison table and tuning results to screenshots/ folder.
Usage: python scripts/tune_and_evaluate.py
"""

import json
import logging
import os
import warnings

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = "data/heart.csv"
OUTPUT_DIR = "screenshots"
RANDOM_STATE = 42
TEST_SIZE = 0.2

NUMERICAL_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]


def build_preprocessor(num_feats, cat_feats):
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
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


def evaluate(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    return {
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1": round(f1_score(y_test, y_pred), 4),
        "ROC-AUC": round(roc_auc_score(y_test, y_proba), 4),
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    df = pd.read_csv(DATA_PATH)
    if "target" not in df.columns:
        df = df.rename(columns={df.columns[-1]: "target"})
    df["target"] = (df["target"] > 0).astype(int)

    num_feats = [f for f in NUMERICAL_FEATURES if f in df.columns]
    cat_feats = [f for f in CATEGORICAL_FEATURES if f in df.columns]

    X = df[num_feats + cat_feats]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    preprocessor = build_preprocessor(num_feats, cat_feats)

    # ── Hyperparameter Tuning ─────────────────────────────────────────────────
    logger.info("Tuning Logistic Regression...")
    lr_pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", LogisticRegression(random_state=RANDOM_STATE, max_iter=1000))])
    lr_params = {"classifier__C": [0.01, 0.1, 1.0, 10.0], "classifier__solver": ["lbfgs", "liblinear"]}
    lr_grid = GridSearchCV(lr_pipeline, lr_params, cv=cv, scoring="roc_auc", n_jobs=-1)
    lr_grid.fit(X_train, y_train)
    logger.info(f"  Best LR params: {lr_grid.best_params_} | CV AUC: {lr_grid.best_score_:.4f}")

    logger.info("Tuning Random Forest...")
    rf_pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", RandomForestClassifier(random_state=RANDOM_STATE))])
    rf_params = {"classifier__n_estimators": [100, 200], "classifier__max_depth": [6, 8, None], "classifier__min_samples_split": [2, 5]}
    rf_grid = GridSearchCV(rf_pipeline, rf_params, cv=cv, scoring="roc_auc", n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    logger.info(f"  Best RF params: {rf_grid.best_params_} | CV AUC: {rf_grid.best_score_:.4f}")

    logger.info("Tuning Gradient Boosting...")
    gb_pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", GradientBoostingClassifier(random_state=RANDOM_STATE))])
    gb_params = {"classifier__n_estimators": [100, 150], "classifier__learning_rate": [0.05, 0.1], "classifier__max_depth": [3, 4]}
    gb_grid = GridSearchCV(gb_pipeline, gb_params, cv=cv, scoring="roc_auc", n_jobs=-1)
    gb_grid.fit(X_train, y_train)
    logger.info(f"  Best GB params: {gb_grid.best_params_} | CV AUC: {gb_grid.best_score_:.4f}")

    # ── Model Comparison Table ────────────────────────────────────────────────
    results = {
        "Logistic Regression": evaluate(lr_grid.best_estimator_, X_test, y_test),
        "Random Forest": evaluate(rf_grid.best_estimator_, X_test, y_test),
        "Gradient Boosting": evaluate(gb_grid.best_estimator_, X_test, y_test),
    }

    comparison_df = pd.DataFrame(results).T
    comparison_df["Best"] = comparison_df["ROC-AUC"] == comparison_df["ROC-AUC"].max()

    logger.info("\n=== MODEL COMPARISON ===")
    print(comparison_df.drop("Best", axis=1).to_string())

    # Save comparison table as image
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")
    table_data = comparison_df.drop("Best", axis=1).reset_index()
    table_data.columns = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]

    colors = []
    best_idx = comparison_df["ROC-AUC"].idxmax()
    for i, row in table_data.iterrows():
        if row["Model"] == best_idx:
            colors.append(["#d5f5e3"] * len(table_data.columns))
        else:
            colors.append(["#f8f9fa"] * len(table_data.columns))

    tbl = ax.table(
        cellText=table_data.values,
        colLabels=table_data.columns,
        cellLoc="center",
        loc="center",
        cellColours=colors,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1.2, 2.5)

    # Header styling
    for j in range(len(table_data.columns)):
        tbl[0, j].set_facecolor("#2c3e50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    ax.set_title("Model Comparison — Heart Disease Classification\n(Best model highlighted in green)", fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "task2_model_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")

    # Save tuning results
    tuning_results = {
        "logistic_regression": {
            "best_params": lr_grid.best_params_,
            "best_cv_auc": round(lr_grid.best_score_, 4),
            "test_metrics": results["Logistic Regression"],
        },
        "random_forest": {
            "best_params": rf_grid.best_params_,
            "best_cv_auc": round(rf_grid.best_score_, 4),
            "test_metrics": results["Random Forest"],
        },
        "gradient_boosting": {
            "best_params": gb_grid.best_params_,
            "best_cv_auc": round(gb_grid.best_score_, 4),
            "test_metrics": results["Gradient Boosting"],
        },
    }
    with open(os.path.join(OUTPUT_DIR, "task2_tuning_results.json"), "w") as f:
        json.dump(tuning_results, f, indent=2)
    logger.info("Saved: screenshots/task2_tuning_results.json")

    # Bar chart comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Model Performance Comparison", fontsize=14, fontweight="bold")

    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    x = range(len(metrics_to_plot))
    width = 0.25
    colors_bar = ["#4C72B0", "#55A868", "#C44E52"]
    model_names = list(results.keys())

    for i, (name, color) in enumerate(zip(model_names, colors_bar)):
        vals = [results[name][m] for m in metrics_to_plot]
        axes[0].bar([xi + i * width for xi in x], vals, width, label=name, color=color, edgecolor="white")

    axes[0].set_xticks([xi + width for xi in x])
    axes[0].set_xticklabels(metrics_to_plot, rotation=15)
    axes[0].set_ylabel("Score")
    axes[0].set_ylim(0.5, 1.05)
    axes[0].legend(fontsize=9)
    axes[0].set_title("All Metrics Comparison")
    axes[0].grid(True, alpha=0.3, axis="y")

    roc_vals = [results[n]["ROC-AUC"] for n in model_names]
    bars = axes[1].bar(model_names, roc_vals, color=colors_bar, edgecolor="white")
    axes[1].set_ylabel("ROC-AUC Score")
    axes[1].set_ylim(0.8, 1.0)
    axes[1].set_title("ROC-AUC Comparison")
    axes[1].grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, roc_vals):
        axes[1].text(bar.get_x() + bar.get_width() / 2, val + 0.002, f"{val:.4f}", ha="center", fontweight="bold")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "task2_model_performance_chart.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")

    logger.info("\nTask 2 complete! Files saved:")
    logger.info("  screenshots/task2_model_comparison.png")
    logger.info("  screenshots/task2_model_performance_chart.png")
    logger.info("  screenshots/task2_tuning_results.json")


if __name__ == "__main__":
    main()
