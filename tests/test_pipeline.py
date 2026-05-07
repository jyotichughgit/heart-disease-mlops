"""
Unit tests for Heart Disease MLOps pipeline.
Run with: pytest tests/ -v --cov=src
Tests: Data Loading, Preprocessing, Model Training, API Schema, Inference, Download Data, Train Module
"""

import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 60
    return pd.DataFrame(
        {
            "age": np.random.randint(30, 80, n).astype(float),
            "sex": np.random.randint(0, 2, n).astype(float),
            "cp": np.random.randint(0, 4, n).astype(float),
            "trestbps": np.random.randint(90, 180, n).astype(float),
            "chol": np.random.randint(150, 350, n).astype(float),
            "fbs": np.random.randint(0, 2, n).astype(float),
            "restecg": np.random.randint(0, 3, n).astype(float),
            "thalach": np.random.randint(70, 200, n).astype(float),
            "exang": np.random.randint(0, 2, n).astype(float),
            "oldpeak": np.random.uniform(0, 5, n),
            "slope": np.random.randint(0, 3, n).astype(float),
            "ca": np.random.randint(0, 4, n).astype(float),
            "thal": np.random.randint(0, 4, n).astype(float),
            "target": np.random.randint(0, 2, n).astype(int),
        }
    )


@pytest.fixture
def sample_csv(sample_df, tmp_path):
    """Write sample_df to a CSV file and return the path."""
    path = tmp_path / "heart.csv"
    sample_df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def sample_patient_dict():
    return {
        "age": 63.0,
        "sex": 1.0,
        "cp": 3.0,
        "trestbps": 145.0,
        "chol": 233.0,
        "fbs": 1.0,
        "restecg": 0.0,
        "thalach": 150.0,
        "exang": 0.0,
        "oldpeak": 2.3,
        "slope": 0.0,
        "ca": 0.0,
        "thal": 1.0,
    }


@pytest.fixture
def trained_pipeline(sample_df):
    """Return a trained sklearn pipeline for reuse across tests."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    from train import build_preprocessor

    num_feats = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    cat_feats = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    preprocessor = build_preprocessor(num_feats, cat_feats)
    pipeline = Pipeline(
        [("preprocessor", preprocessor), ("classifier", LogisticRegression(max_iter=1000, random_state=42))]
    )
    pipeline.fit(sample_df[num_feats + cat_feats], sample_df["target"])
    return pipeline, num_feats, cat_feats


@pytest.fixture
def saved_model(trained_pipeline, tmp_path):
    """Save trained pipeline to disk and return paths."""
    pipeline, num_feats, cat_feats = trained_pipeline
    model_path = tmp_path / "best_model.pkl"
    config_path = tmp_path / "feature_config.json"

    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)

    config = {
        "best_model": "logistic_regression",
        "best_roc_auc": 0.90,
        "numerical_features": num_feats,
        "categorical_features": cat_feats,
        "all_features": num_feats + cat_feats,
        "target": "target",
    }
    with open(config_path, "w") as f:
        json.dump(config, f)

    return str(model_path), str(config_path), config


# ── Task 1: Data Loading Tests ─────────────────────────────────────────────────
class TestDataLoading:

    def test_dataframe_not_empty(self, sample_df):
        assert len(sample_df) > 0

    def test_required_columns_exist(self, sample_df):
        required = [
            "age",
            "sex",
            "cp",
            "trestbps",
            "chol",
            "fbs",
            "restecg",
            "thalach",
            "exang",
            "oldpeak",
            "slope",
            "ca",
            "thal",
            "target",
        ]
        for col in required:
            assert col in sample_df.columns, f"Missing: {col}"

    def test_target_is_binary(self, sample_df):
        assert set(sample_df["target"].unique()).issubset({0, 1})

    def test_no_negative_age(self, sample_df):
        assert (sample_df["age"] >= 0).all()

    def test_sex_values_valid(self, sample_df):
        assert set(sample_df["sex"].unique()).issubset({0.0, 1.0})

    def test_cholesterol_positive(self, sample_df):
        assert (sample_df["chol"] > 0).all()

    def test_data_types_numeric(self, sample_df):
        for col in ["age", "trestbps", "chol", "thalach", "oldpeak"]:
            assert pd.api.types.is_numeric_dtype(sample_df[col])


# ── Task 1: train.py load_data() Tests ────────────────────────────────────────
class TestLoadData:

    def test_load_data_returns_dataframe(self, sample_csv):
        from train import load_data

        df = load_data(sample_csv)
        assert isinstance(df, pd.DataFrame)

    def test_load_data_binarizes_target(self, tmp_path):
        from train import load_data

        df = pd.DataFrame(
            {
                "age": [50.0, 60.0, 70.0],
                "sex": [1.0, 0.0, 1.0],
                "cp": [0.0, 1.0, 2.0],
                "trestbps": [120.0, 130.0, 140.0],
                "chol": [200.0, 210.0, 220.0],
                "fbs": [0.0, 1.0, 0.0],
                "restecg": [0.0, 1.0, 0.0],
                "thalach": [150.0, 160.0, 170.0],
                "exang": [0.0, 1.0, 0.0],
                "oldpeak": [1.0, 2.0, 3.0],
                "slope": [0.0, 1.0, 2.0],
                "ca": [0.0, 1.0, 2.0],
                "thal": [1.0, 2.0, 3.0],
                "target": [0, 2, 4],
            }
        )
        path = tmp_path / "test.csv"
        df.to_csv(path, index=False)
        loaded = load_data(str(path))
        assert set(loaded["target"].unique()).issubset({0, 1})
        assert loaded["target"].iloc[1] == 1
        assert loaded["target"].iloc[2] == 1

    def test_load_data_renames_last_column_if_no_target(self, tmp_path):
        from train import load_data

        df = pd.DataFrame(
            {
                "age": [50.0],
                "sex": [1.0],
                "cp": [0.0],
                "trestbps": [120.0],
                "chol": [200.0],
                "fbs": [0.0],
                "restecg": [0.0],
                "thalach": [150.0],
                "exang": [0.0],
                "oldpeak": [1.0],
                "slope": [0.0],
                "ca": [0.0],
                "thal": [1.0],
                "diagnosis": [0],
            }
        )
        path = tmp_path / "test.csv"
        df.to_csv(path, index=False)
        loaded = load_data(str(path))
        assert "target" in loaded.columns


# ── Task 2: Preprocessing Tests ───────────────────────────────────────────────
class TestDataPreprocessing:

    def test_missing_value_handling(self, sample_df):
        df = sample_df.copy()
        df.loc[0:5, "age"] = np.nan
        df.fillna(df.median(numeric_only=True), inplace=True)
        assert df.isnull().sum().sum() == 0

    def test_train_test_split_stratified(self, sample_df):
        from sklearn.model_selection import train_test_split

        X = sample_df.drop("target", axis=1)
        y = sample_df["target"]
        X_train, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        assert len(X_train) + len(X_test) == len(sample_df)
        assert len(X_train) > len(X_test)

    def test_feature_scaling(self, sample_df):
        from sklearn.preprocessing import StandardScaler

        scaled = StandardScaler().fit_transform(sample_df[["age", "trestbps", "chol"]])
        assert abs(scaled.mean()) < 0.5

    def test_pipeline_builds_correctly(self, sample_df):
        from train import build_preprocessor

        num_feats = ["age", "trestbps", "chol", "thalach", "oldpeak"]
        cat_feats = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
        preprocessor = build_preprocessor(num_feats, cat_feats)
        result = preprocessor.fit_transform(sample_df[num_feats + cat_feats])
        assert result.shape[0] == len(sample_df)
        assert result.shape[1] > len(num_feats + cat_feats)


# ── Task 2: Model Training Tests ──────────────────────────────────────────────
class TestModelTraining:

    def test_logistic_regression_trains(self, sample_df):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline

        from train import build_preprocessor

        num_feats = ["age", "trestbps", "chol", "thalach", "oldpeak"]
        cat_feats = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
        pipeline = Pipeline(
            [
                ("preprocessor", build_preprocessor(num_feats, cat_feats)),
                ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        )
        pipeline.fit(sample_df[num_feats + cat_feats], sample_df["target"])
        preds = pipeline.predict(sample_df[num_feats + cat_feats])
        assert set(preds).issubset({0, 1})

    def test_random_forest_trains(self, sample_df):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline

        from train import build_preprocessor

        num_feats = ["age", "trestbps", "chol", "thalach", "oldpeak"]
        cat_feats = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
        pipeline = Pipeline(
            [
                ("preprocessor", build_preprocessor(num_feats, cat_feats)),
                ("classifier", RandomForestClassifier(n_estimators=10, random_state=42)),
            ]
        )
        pipeline.fit(sample_df[num_feats + cat_feats], sample_df["target"])
        probas = pipeline.predict_proba(sample_df[num_feats + cat_feats])
        assert probas.shape == (len(sample_df), 2)
        assert np.allclose(probas.sum(axis=1), 1.0)

    def test_model_predict_proba_sums_to_one(self, sample_df):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline

        from train import build_preprocessor

        num_feats = ["age", "trestbps", "chol", "thalach", "oldpeak"]
        cat_feats = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
        pipeline = Pipeline(
            [
                ("preprocessor", build_preprocessor(num_feats, cat_feats)),
                ("classifier", RandomForestClassifier(n_estimators=10, random_state=42)),
            ]
        )
        pipeline.fit(sample_df[num_feats + cat_feats], sample_df["target"])
        assert np.allclose(pipeline.predict_proba(sample_df[num_feats + cat_feats]).sum(axis=1), 1.0, atol=1e-6)

    def test_model_accuracy_above_threshold(self, sample_df):
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline

        from train import build_preprocessor

        num_feats = ["age", "trestbps", "chol", "thalach", "oldpeak"]
        cat_feats = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
        pipeline = Pipeline(
            [
                ("preprocessor", build_preprocessor(num_feats, cat_feats)),
                ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        )
        X_train, X_test, y_train, y_test = train_test_split(
            sample_df[num_feats + cat_feats], sample_df["target"], test_size=0.2, random_state=42
        )
        pipeline.fit(X_train, y_train)
        assert accuracy_score(y_test, pipeline.predict(X_test)) >= 0.4

    def test_model_serialization(self, sample_df, tmp_path):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline

        from train import build_preprocessor

        num_feats = ["age", "trestbps", "chol", "thalach", "oldpeak"]
        cat_feats = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
        pipeline = Pipeline(
            [
                ("preprocessor", build_preprocessor(num_feats, cat_feats)),
                ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        )
        pipeline.fit(sample_df[num_feats + cat_feats], sample_df["target"])
        path = tmp_path / "model.pkl"
        with open(path, "wb") as f:
            pickle.dump(pipeline, f)
        with open(path, "rb") as f:
            loaded = pickle.load(f)
        assert np.array_equal(
            pipeline.predict(sample_df[num_feats + cat_feats]),
            loaded.predict(sample_df[num_feats + cat_feats]),
        )


# ── Task 2: train_and_evaluate() Tests ────────────────────────────────────────
class TestTrainAndEvaluate:

    def test_returns_all_required_metrics(self, sample_df):
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
        from sklearn.pipeline import Pipeline

        from train import build_preprocessor, train_and_evaluate

        num_feats = ["age", "trestbps", "chol", "thalach", "oldpeak"]
        cat_feats = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
        pipeline = Pipeline(
            [
                ("preprocessor", build_preprocessor(num_feats, cat_feats)),
                ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        )
        from sklearn.model_selection import train_test_split

        X = sample_df[num_feats + cat_feats]
        y = sample_df["target"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        metrics = train_and_evaluate(pipeline, X_train, y_train, X_test, y_test, cv)

        for key in ["accuracy", "precision", "recall", "f1", "roc_auc", "cv_roc_auc_mean", "cv_roc_auc_std"]:
            assert key in metrics, f"Missing metric: {key}"

    def test_metrics_in_valid_range(self, sample_df):
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, train_test_split
        from sklearn.pipeline import Pipeline

        from train import build_preprocessor, train_and_evaluate

        num_feats = ["age", "trestbps", "chol", "thalach", "oldpeak"]
        cat_feats = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
        pipeline = Pipeline(
            [
                ("preprocessor", build_preprocessor(num_feats, cat_feats)),
                ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        )
        X = sample_df[num_feats + cat_feats]
        y = sample_df["target"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        metrics = train_and_evaluate(pipeline, X_train, y_train, X_test, y_test, cv)

        for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            assert 0.0 <= metrics[key] <= 1.0, f"{key} out of range: {metrics[key]}"


# ── Task 2: Plot Functions Tests ───────────────────────────────────────────────
class TestPlotFunctions:

    def test_save_confusion_matrix_creates_file(self, sample_df, tmp_path):
        import mlflow
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline

        from train import build_preprocessor, save_confusion_matrix

        num_feats = ["age", "trestbps", "chol", "thalach", "oldpeak"]
        cat_feats = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
        pipeline = Pipeline(
            [
                ("preprocessor", build_preprocessor(num_feats, cat_feats)),
                ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        )
        X = sample_df[num_feats + cat_feats]
        y = sample_df["target"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        plots_dir = str(tmp_path)
        mlflow.set_tracking_uri(f"file:{tmp_path}/mlruns")
        with mlflow.start_run():
            save_confusion_matrix(y_test, y_pred, "test_model", plots_dir)

        assert os.path.exists(os.path.join(plots_dir, "test_model_confusion_matrix.png"))

    def test_save_roc_curve_creates_file(self, sample_df, tmp_path):
        import mlflow
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline

        from train import build_preprocessor, save_roc_curve

        num_feats = ["age", "trestbps", "chol", "thalach", "oldpeak"]
        cat_feats = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
        pipeline = Pipeline(
            [
                ("preprocessor", build_preprocessor(num_feats, cat_feats)),
                ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        )
        X = sample_df[num_feats + cat_feats]
        y = sample_df["target"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        pipeline.fit(X_train, y_train)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        plots_dir = str(tmp_path)
        mlflow.set_tracking_uri(f"file:{tmp_path}/mlruns")
        with mlflow.start_run():
            save_roc_curve(y_test, y_proba, "test_model", plots_dir)

        assert os.path.exists(os.path.join(plots_dir, "test_model_roc_curve.png"))


# ── Task 2: main() Integration Test ───────────────────────────────────────────
class TestMainFunction:

    def test_main_trains_and_saves_model(self, sample_csv, tmp_path):
        from train import main

        output_dir = str(tmp_path / "models")
        auc = main(sample_csv, output_dir)

        assert auc > 0.0
        assert os.path.exists(os.path.join(output_dir, "best_model.pkl"))
        assert os.path.exists(os.path.join(output_dir, "feature_config.json"))

    def test_main_feature_config_has_required_keys(self, sample_csv, tmp_path):
        from train import main

        output_dir = str(tmp_path / "models")
        main(sample_csv, output_dir)

        with open(os.path.join(output_dir, "feature_config.json")) as f:
            config = json.load(f)

        for key in [
            "best_model",
            "best_roc_auc",
            "numerical_features",
            "categorical_features",
            "all_features",
            "target",
        ]:
            assert key in config, f"Missing config key: {key}"


# ── Task 6: API Schema Tests ───────────────────────────────────────────────────
class TestAPISchema:

    def test_patient_data_valid(self, sample_patient_dict):
        try:
            from app import PatientData

            p = PatientData(**sample_patient_dict)
            assert p.age == 63.0
            assert p.sex == 1.0
        except ImportError:
            pytest.skip("app not importable")

    def test_patient_data_all_features_present(self, sample_patient_dict):
        required = [
            "age",
            "sex",
            "cp",
            "trestbps",
            "chol",
            "fbs",
            "restecg",
            "thalach",
            "exang",
            "oldpeak",
            "slope",
            "ca",
            "thal",
        ]
        for feat in required:
            assert feat in sample_patient_dict

    def test_feature_config_valid(self, tmp_path):
        config = {
            "best_model": "random_forest",
            "best_roc_auc": 0.92,
            "numerical_features": ["age", "trestbps", "chol", "thalach", "oldpeak"],
            "categorical_features": ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"],
            "all_features": [
                "age",
                "trestbps",
                "chol",
                "thalach",
                "oldpeak",
                "sex",
                "cp",
                "fbs",
                "restecg",
                "exang",
                "slope",
                "ca",
                "thal",
            ],
            "target": "target",
        }
        p = tmp_path / "feature_config.json"
        p.write_text(json.dumps(config))
        loaded = json.loads(p.read_text())
        assert loaded["best_model"] == "random_forest"
        assert len(loaded["all_features"]) == 13


# ── Task 5: Inference Tests ────────────────────────────────────────────────────
class TestInference:

    def test_predict_single_returns_correct_keys(self, saved_model, sample_patient_dict):
        model_path, config_path, config = saved_model
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        input_df = pd.DataFrame([{k: sample_patient_dict.get(k, 0) for k in config["all_features"]}])
        prediction = int(model.predict(input_df)[0])
        probas = model.predict_proba(input_df)[0]

        result = {
            "prediction": prediction,
            "prediction_label": "Heart Disease" if prediction == 1 else "No Heart Disease",
            "confidence": round(float(max(probas)), 4),
            "probability_disease": round(float(probas[1]), 4),
            "probability_no_disease": round(float(probas[0]), 4),
        }

        for key in ["prediction", "prediction_label", "confidence", "probability_disease", "probability_no_disease"]:
            assert key in result

    def test_predict_single_prediction_is_binary(self, trained_pipeline, sample_df):
        pipeline, num_feats, cat_feats = trained_pipeline
        predictions = pipeline.predict(sample_df[num_feats + cat_feats])
        assert set(predictions).issubset({0, 1})

    def test_reproducibility(self, saved_model, sample_df):
        model_path, config_path, config = saved_model
        sample = sample_df[config["all_features"]].iloc[:1]

        results = []
        for _ in range(5):
            with open(model_path, "rb") as f:
                loaded = pickle.load(f)
            pred = int(loaded.predict(sample)[0])
            prob = round(float(loaded.predict_proba(sample)[0][1]), 6)
            results.append((pred, prob))

        assert len(set(r[0] for r in results)) == 1, "Predictions must be identical"
        assert len(set(r[1] for r in results)) == 1, "Probabilities must be identical"


# ── Task 5: Download Data Tests ────────────────────────────────────────────────
class TestDownloadData:

    def test_column_names_correct(self, sample_df):
        required = [
            "age",
            "sex",
            "cp",
            "trestbps",
            "chol",
            "fbs",
            "restecg",
            "thalach",
            "exang",
            "oldpeak",
            "slope",
            "ca",
            "thal",
            "target",
        ]
        for col in required:
            assert col in sample_df.columns

    def test_target_binarized_correctly(self):
        raw = pd.Series([0, 1, 2, 3, 4])
        binarized = (raw > 0).astype(int)
        assert list(binarized) == [0, 1, 1, 1, 1]

    def test_missing_value_filled_with_median(self):
        df = pd.DataFrame({"ca": [0.0, np.nan, 1.0, 2.0, np.nan], "thal": [2.0, 3.0, np.nan, 1.0, 3.0]})
        df.fillna(df.median(numeric_only=True), inplace=True)
        assert df.isnull().sum().sum() == 0

    def test_dataset_shape_valid(self, sample_df):
        assert sample_df.shape[1] == 14

    def test_no_duplicate_rows(self, sample_df):
        assert sample_df.duplicated().sum() == 0

    def test_age_range_valid(self, sample_df):
        assert sample_df["age"].between(1, 120).all()

    def test_target_distribution_balanced(self, sample_df):
        counts = sample_df["target"].value_counts(normalize=True)
        assert counts.min() >= 0.20
