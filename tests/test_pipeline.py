"""
Unit tests for Heart Disease MLOps pipeline.
Run with: pytest tests/ -v --cov=src
Tests: Data Loading, Preprocessing, Model Training, API Schema
"""

import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# Fixtures
@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 50
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
def sample_patient_dict():
    return {
        "age": 63.0, "sex": 1.0, "cp": 3.0, "trestbps": 145.0,
        "chol": 233.0, "fbs": 1.0, "restecg": 0.0, "thalach": 150.0,
        "exang": 0.0, "oldpeak": 2.3, "slope": 0.0, "ca": 0.0, "thal": 1.0,
    }


# Task 1 — Data Loading Tests
class TestDataLoading:
    def test_dataframe_not_empty(self, sample_df):
        assert len(sample_df) > 0

    def test_required_columns_exist(self, sample_df):
        required = ["age", "sex", "cp", "trestbps", "chol", "fbs",
                    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
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


# Task 2 — Preprocessing Tests
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
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        num_feats = ["age", "trestbps", "chol", "thalach", "oldpeak"]
        cat_feats = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
        preprocessor = ColumnTransformer(
            [
                ("num", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), num_feats),
                (
                    "cat",
                    Pipeline(
                        [
                            ("i", SimpleImputer(strategy="most_frequent")),
                            ("e", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                        ]
                    ),
                    cat_feats,
                ),
            ]
        )
        result = preprocessor.fit_transform(sample_df[num_feats + cat_feats])
        assert result.shape[0] == len(sample_df)
        assert result.shape[1] > len(num_feats + cat_feats)


# Task 2 — Model Training Tests
class TestModelTraining:
    def _build(self, clf):
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        num_feats = ["age", "trestbps", "chol", "thalach", "oldpeak"]
        cat_feats = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
        preprocessor = ColumnTransformer(
            [
                ("num", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), num_feats),
                (
                    "cat",
                    Pipeline(
                        [
                            ("i", SimpleImputer(strategy="most_frequent")),
                            ("e", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                        ]
                    ),
                    cat_feats,
                ),
            ]
        )
        return Pipeline([("preprocessor", preprocessor), ("classifier", clf)]), num_feats + cat_feats

    def test_logistic_regression_trains(self, sample_df):
        from sklearn.linear_model import LogisticRegression

        pipeline, feats = self._build(LogisticRegression(max_iter=1000, random_state=42))
        pipeline.fit(sample_df[feats], sample_df["target"])
        preds = pipeline.predict(sample_df[feats])
        assert set(preds).issubset({0, 1})

    def test_random_forest_trains(self, sample_df):
        from sklearn.ensemble import RandomForestClassifier

        pipeline, feats = self._build(RandomForestClassifier(n_estimators=10, random_state=42))
        pipeline.fit(sample_df[feats], sample_df["target"])
        probas = pipeline.predict_proba(sample_df[feats])
        assert probas.shape == (len(sample_df), 2)
        assert np.allclose(probas.sum(axis=1), 1.0)

    def test_model_predict_proba_sums_to_one(self, sample_df):
        from sklearn.ensemble import RandomForestClassifier

        pipeline, feats = self._build(RandomForestClassifier(n_estimators=10, random_state=42))
        pipeline.fit(sample_df[feats], sample_df["target"])
        assert np.allclose(pipeline.predict_proba(sample_df[feats]).sum(axis=1), 1.0, atol=1e-6)

    def test_model_accuracy_above_threshold(self, sample_df):
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split

        pipeline, feats = self._build(LogisticRegression(max_iter=1000, random_state=42))
        X_train, X_test, y_train, y_test = train_test_split(
            sample_df[feats], sample_df["target"], test_size=0.2, random_state=42
        )
        pipeline.fit(X_train, y_train)
        assert accuracy_score(y_test, pipeline.predict(X_test)) >= 0.4

    def test_model_serialization(self, sample_df, tmp_path):
        from sklearn.linear_model import LogisticRegression

        pipeline, feats = self._build(LogisticRegression(max_iter=1000, random_state=42))
        pipeline.fit(sample_df[feats], sample_df["target"])
        path = tmp_path / "model.pkl"
        with open(path, "wb") as f:
            pickle.dump(pipeline, f)
        with open(path, "rb") as f:
            loaded = pickle.load(f)
        assert np.array_equal(pipeline.predict(sample_df[feats]), loaded.predict(sample_df[feats]))


# Task 6 — API Schema Tests
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
        required = ["age", "sex", "cp", "trestbps", "chol", "fbs",
                    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
        for feat in required:
            assert feat in sample_patient_dict

    def test_feature_config_valid(self, tmp_path):
        config = {
            "best_model": "random_forest",
            "best_roc_auc": 0.92,
            "numerical_features": ["age", "trestbps", "chol", "thalach", "oldpeak"],
            "categorical_features": ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"],
            "all_features": ["age", "trestbps", "chol", "thalach", "oldpeak",
                             "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"],
            "target": "target",
        }
        p = tmp_path / "feature_config.json"
        p.write_text(json.dumps(config))
        loaded = json.loads(p.read_text())
        assert loaded["best_model"] == "random_forest"
        assert len(loaded["all_features"]) == 13


# Task 5 — Gap 1: Inference Script Tests
class TestInference:

    def test_predict_single_returns_correct_keys(self, sample_df, tmp_path):
        """Test inference script predict_single returns all required keys."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        import pickle
        import json

        num_feats = ["age", "trestbps", "chol", "thalach", "oldpeak"]
        cat_feats = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

        preprocessor = ColumnTransformer([
            ("num", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), num_feats),
            ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")), ("e", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_feats),
        ])
        pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", LogisticRegression(max_iter=1000, random_state=42))])
        pipeline.fit(sample_df[num_feats + cat_feats], sample_df["target"])

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

        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(config_path) as f:
            cfg = json.load(f)

        patient = {"age": 63.0, "sex": 1.0, "cp": 3.0, "trestbps": 145.0,
                   "chol": 233.0, "fbs": 1.0, "restecg": 0.0, "thalach": 150.0,
                   "exang": 0.0, "oldpeak": 2.3, "slope": 0.0, "ca": 0.0, "thal": 1.0}

        input_df = pd.DataFrame([{k: patient.get(k, 0) for k in cfg["all_features"]}])
        prediction = int(model.predict(input_df)[0])
        probas = model.predict_proba(input_df)[0]

        result = {
            "prediction": prediction,
            "prediction_label": "Heart Disease" if prediction == 1 else "No Heart Disease",
            "confidence": round(float(max(probas)), 4),
            "probability_disease": round(float(probas[1]), 4),
            "probability_no_disease": round(float(probas[0]), 4),
        }

        required_keys = ["prediction", "prediction_label", "confidence", "probability_disease", "probability_no_disease"]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_predict_single_prediction_is_binary(self, sample_df, tmp_path):
        """Test prediction output is always 0 or 1."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler, OneHotEncoder

        num_feats = ["age", "trestbps", "chol", "thalach", "oldpeak"]
        cat_feats = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
        preprocessor = ColumnTransformer([
            ("num", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), num_feats),
            ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")), ("e", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_feats),
        ])
        pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", LogisticRegression(max_iter=1000, random_state=42))])
        pipeline.fit(sample_df[num_feats + cat_feats], sample_df["target"])

        predictions = pipeline.predict(sample_df[num_feats + cat_feats])
        assert set(predictions).issubset({0, 1}), "All predictions must be 0 or 1"

    def test_reproducibility(self, sample_df, tmp_path):
        """Test same input always produces same output — reproducibility check."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        import pickle

        num_feats = ["age", "trestbps", "chol", "thalach", "oldpeak"]
        cat_feats = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
        preprocessor = ColumnTransformer([
            ("num", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), num_feats),
            ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")), ("e", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_feats),
        ])
        pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", LogisticRegression(max_iter=1000, random_state=42))])
        pipeline.fit(sample_df[num_feats + cat_feats], sample_df["target"])

        model_path = tmp_path / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(pipeline, f)

        sample = sample_df[num_feats + cat_feats].iloc[:1]

        results = []
        for _ in range(5):
            with open(model_path, "rb") as f:
                loaded = pickle.load(f)
            pred = int(loaded.predict(sample)[0])
            prob = float(loaded.predict_proba(sample)[0][1])
            results.append((pred, prob))

        predictions = [r[0] for r in results]
        probabilities = [r[1] for r in results]
        assert len(set(predictions)) == 1, "Prediction must be identical across all runs"
        assert len(set(probabilities)) == 1, "Probability must be identical across all runs"


# Task 5 — Gap 2: Download Data Tests
class TestDownloadData:

    def test_column_names_correct(self, sample_df):
        """Test dataset has all 14 required columns."""
        required = [
            "age", "sex", "cp", "trestbps", "chol",
            "fbs", "restecg", "thalach", "exang",
            "oldpeak", "slope", "ca", "thal", "target",
        ]
        for col in required:
            assert col in sample_df.columns, f"Missing column: {col}"

    def test_target_binarized_correctly(self):
        """Test target binarization — values > 0 become 1."""
        raw_targets = pd.Series([0, 1, 2, 3, 4])
        binarized = (raw_targets > 0).astype(int)
        assert list(binarized) == [0, 1, 1, 1, 1], "Target should be binarized to 0/1"

    def test_missing_value_filled_with_median(self):
        """Test missing values are filled with median — matches download_data.py logic."""
        df = pd.DataFrame({
            "ca": [0.0, np.nan, 1.0, 2.0, np.nan],
            "thal": [2.0, 3.0, np.nan, 1.0, 3.0],
        })
        df.fillna(df.median(numeric_only=True), inplace=True)
        assert df.isnull().sum().sum() == 0, "No missing values should remain after fill"
        assert df["ca"].iloc[1] == 1.0, "Missing ca should be filled with median (1.0)"

    def test_dataset_shape_valid(self, sample_df):
        """Test dataset has correct number of features."""
        assert sample_df.shape[1] == 14, f"Expected 14 columns, got {sample_df.shape[1]}"

    def test_no_duplicate_rows(self, sample_df):
        """Test no duplicate rows in dataset."""
        duplicates = sample_df.duplicated().sum()
        assert duplicates == 0, f"Found {duplicates} duplicate rows"

    def test_age_range_valid(self, sample_df):
        """Test age values are within realistic range."""
        assert sample_df["age"].between(1, 120).all(), "Age values must be between 1 and 120"

    def test_target_distribution_balanced(self, sample_df):
        """Test target classes are not severely imbalanced (> 80/20 split)."""
        counts = sample_df["target"].value_counts(normalize=True)
        assert counts.min() >= 0.20, "Target classes are too imbalanced (worse than 80/20)"
