"""
scripts/eda.py
Performs EDA on Heart Disease dataset and saves professional visualizations.
Usage: python scripts/eda.py
Outputs: screenshots/eda_*.png
"""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = "data/heart.csv"
OUTPUT_DIR = "screenshots"

# Feature descriptions for better plot labels
FEATURE_DESCRIPTIONS = {
    "age": "Age (years)",
    "sex": "Sex (1=Male, 0=Female)",
    "cp": "Chest Pain Type (0-3)",
    "trestbps": "Resting Blood Pressure (mmHg)",
    "chol": "Serum Cholesterol (mg/dl)",
    "fbs": "Fasting Blood Sugar >120 (1=Yes)",
    "restecg": "Resting ECG Results (0-2)",
    "thalach": "Max Heart Rate Achieved",
    "exang": "Exercise Induced Angina (1=Yes)",
    "oldpeak": "ST Depression (Exercise)",
    "slope": "ST Slope (0-2)",
    "ca": "Major Vessels (0-3)",
    "thal": "Thalassemia Type",
    "target": "Heart Disease (1=Yes)",
}


def load_data():
    logger.info(f"Loading data from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    if "target" not in df.columns:
        df = df.rename(columns={df.columns[-1]: "target"})
    df["target"] = (df["target"] > 0).astype(int)
    logger.info(f"Shape: {df.shape} | Target: {df['target'].value_counts().to_dict()}")
    return df


def plot_class_balance(df):
    """Plot 1 — Class distribution (bar + pie)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Class Distribution — Heart Disease Dataset", fontsize=15, fontweight="bold", y=1.02)

    counts = df["target"].value_counts().sort_index()
    labels = ["No Disease (0)", "Disease (1)"]
    colors = ["#4C72B0", "#DD8452"]

    # Bar chart
    bars = axes[0].bar(labels, counts.values, color=colors, edgecolor="black", width=0.5)
    axes[0].set_title("Class Count", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Number of Patients")
    axes[0].set_ylim(0, max(counts.values) * 1.2)
    for bar, val in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, val + 2, str(val), ha="center", fontweight="bold", fontsize=12)

    # Pie chart
    axes[1].pie(
        counts.values,
        labels=labels,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        explode=(0.05, 0.05),
        textprops={"fontsize": 12},
    )
    axes[1].set_title("Class Balance (%)", fontsize=13, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "eda_01_class_balance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")


def plot_feature_histograms(df):
    """Plot 2 — Histograms of all features."""
    cols = [c for c in df.columns if c != "target"]
    n_cols = 4
    n_rows = (len(cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))
    axes = axes.flatten()
    fig.suptitle("Feature Distributions — Heart Disease Dataset", fontsize=16, fontweight="bold")

    for i, col in enumerate(cols):
        axes[i].hist(df[col], bins=20, color="#4C72B0", edgecolor="white", alpha=0.85)
        axes[i].set_title(FEATURE_DESCRIPTIONS.get(col, col), fontsize=10, fontweight="bold")
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Frequency")
        axes[i].grid(True, alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "eda_02_feature_histograms.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")


def plot_correlation_heatmap(df):
    """Plot 3 — Correlation heatmap."""
    fig, ax = plt.subplots(figsize=(14, 11))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 9},
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=15, fontweight="bold", pad=15)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "eda_03_correlation_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")


def plot_boxplots_by_target(df):
    """Plot 4 — Feature distribution by target class."""
    continuous = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    available = [c for c in continuous if c in df.columns]

    fig, axes = plt.subplots(1, len(available), figsize=(18, 6))
    fig.suptitle("Feature Distribution by Target Class", fontsize=14, fontweight="bold")

    for i, col in enumerate(available):
        data0 = df[df["target"] == 0][col]
        data1 = df[df["target"] == 1][col]
        bp = axes[i].boxplot(
            [data0, data1],
            patch_artist=True,
            labels=["No Disease", "Disease"],
            medianprops=dict(color="red", linewidth=2),
        )
        bp["boxes"][0].set_facecolor("#4C72B0")
        bp["boxes"][1].set_facecolor("#DD8452")
        axes[i].set_title(FEATURE_DESCRIPTIONS.get(col, col), fontsize=10, fontweight="bold")
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "eda_04_boxplots_by_target.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")


def plot_missing_values(df):
    """Plot 5 — Missing values analysis."""
    # Load raw data to show original missing values
    raw_cols = [
        "age", "sex", "cp", "trestbps", "chol",
        "fbs", "restecg", "thalach", "exang",
        "oldpeak", "slope", "ca", "thal", "target",
    ]
    missing = df.isnull().sum()
    missing = missing[missing >= 0]  # show all including zeros

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#DD8452" if v > 0 else "#4C72B0" for v in missing.values]
    bars = ax.bar(missing.index, missing.values, color=colors, edgecolor="black")
    ax.set_title("Missing Values per Feature (after cleaning = 0)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Missing Count")
    ax.set_xticklabels(missing.index, rotation=45, ha="right")
    ax.grid(True, alpha=0.3, axis="y")

    for bar, val in zip(bars, missing.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.1, str(val), ha="center", fontsize=9)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "eda_05_missing_values.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")


def plot_feature_vs_target(df):
    """Plot 6 — Categorical features vs target."""
    cat_feats = ["sex", "cp", "fbs", "restecg", "exang", "slope"]
    available = [c for c in cat_feats if c in df.columns]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    fig.suptitle("Categorical Features vs Heart Disease", fontsize=14, fontweight="bold")

    for i, col in enumerate(available):
        ct = pd.crosstab(df[col], df["target"])
        ct.plot(
            kind="bar",
            ax=axes[i],
            color=["#4C72B0", "#DD8452"],
            edgecolor="black",
            rot=0,
        )
        axes[i].set_title(FEATURE_DESCRIPTIONS.get(col, col), fontsize=10, fontweight="bold")
        axes[i].set_xlabel("")
        axes[i].set_ylabel("Count")
        axes[i].legend(["No Disease", "Disease"], fontsize=8)
        axes[i].grid(True, alpha=0.3, axis="y")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "eda_06_categorical_vs_target.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")


def print_summary(df):
    """Print dataset summary statistics."""
    logger.info("\n" + "=" * 60)
    logger.info("DATASET SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total samples:     {len(df)}")
    logger.info(f"Total features:    {len(df.columns) - 1}")
    logger.info(f"No Disease (0):    {(df['target'] == 0).sum()} ({(df['target'] == 0).mean():.1%})")
    logger.info(f"Disease (1):       {(df['target'] == 1).sum()} ({(df['target'] == 1).mean():.1%})")
    logger.info(f"Missing values:    {df.isnull().sum().sum()}")
    logger.info(f"Duplicate rows:    {df.duplicated().sum()}")
    logger.info("\nStatistical Summary:")
    print(df.describe().T.to_string())
    logger.info("=" * 60)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams["figure.dpi"] = 120

    df = load_data()
    print_summary(df)

    logger.info("Generating EDA plots...")
    plot_class_balance(df)
    plot_feature_histograms(df)
    plot_correlation_heatmap(df)
    plot_boxplots_by_target(df)
    plot_missing_values(df)
    plot_feature_vs_target(df)

    logger.info(f"\nAll EDA plots saved to {OUTPUT_DIR}/")
    logger.info("Files saved:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.startswith("eda_"):
            logger.info(f"  {f}")


if __name__ == "__main__":
    main()
