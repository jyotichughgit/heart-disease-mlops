"""
Unit tests for Heart Disease MLOps pipeline.
Run with: pytest tests/ -v --tb=short
"""

import os
import sys
import json
import pickle
import pytest
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Minimal heart disease dataframe for testing."""
    np.random.seed(42)
    n = 50
    return pd.DataFrame({
        "age":      np.random.randint(30, 80, n).astype(float),
        "sex":      np.random.randint(0, 2, n).astype(float),
        "cp":       np.random.randint(0, 4, n).astype(float),
        "trestbps": np.random.randint(90, 180, n).astype(float),
        "chol":     np.random.randint(150, 350, n).astype(float),
        "fbs":      np.random.randint(0, 2, n).astype(float),
        "restecg":  np.random.randint(0, 3, n).astype(float),
        "thalach":  np.random.randint(70, 200, n).astype(float),
        "exang":    np.random.randint(0, 2, n).astype(float),
        "oldpeak":  np.random.uniform(0, 5, n),
        "slope":    np.random.randint(0, 3, n).astype(float),
        "ca":       np.random.randint(0, 4, n).astype(float),
        "thal":     np.random.randint(0, 4, n).astype(float),
        "target":   np.random.randint(0, 2, n).astype(int),
    })


@pytest.fixture
def sample_patient_dict():
    return {
        "age": 63.0, "sex": 1.0, "cp": 3.0, "trestbps": 145.0,
        "chol": 233.0, "fbs": 1.0, "restecg": 0.0, "thalach": 150.0,
        "exang": 0.0, "oldpeak": 2.3, "slope": 0.0, "ca": 0.0, "thal": 1.0
    }


# ─────────────────────────────────────────────────────────────────────────────
# Data Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDataLoading:

    def test_dataframe_not_empty(self, sample_df):
        assert len(sample_df) > 0, "DataFrame should not be empty"

    def test_required_columns_exist(self, sample_df):
        required = ["age", "sex", "cp", "trestbps", "chol", "fbs",
                    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
        for col in required:
            assert col in sample_df.columns, f"Missing column: {col}"

    def test_target_is_binary(self, sample_df):
        unique_vals = set(sample_df["target"].unique())
        assert unique_vals.issubset({0, 1}), f"Target should be binary, got {unique_vals}"

    def test_no_negative_age(self, sample_df):
        assert (sample_df["age"] >= 0).all(), "Age values should be non-negative"

    def test_sex_values_valid(self, sample_df):
        assert set(sample_df["sex"].unique()).issubset({0.0, 1.0}), "Sex should be 0 or 1"

    def test_cholesterol_positive(self, sample_df):
        assert (sample_df["chol"] > 0).all(), "Cholesterol should be positive"

    def test_data_types_numeric(self, sample_df):
        numeric_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
        for col in numeric_cols:
            assert pd.api.types.is_numeric_dtype(sample_df[col]), f"{col} should be numeric"


class TestDataPreprocessing:

    def test_missing_value_handling(self, sample_df):
        df_copy = sample_df.copy()
        df_copy.loc[0:5, "age"]  = np.nan
        df_copy.loc[2:4, "chol"] = np.nan
        df_copy.fillna(df_copy.median(numeric_only=True), inplace=True)
        assert df_copy.isnull().sum().sum() == 0, "No missing values should remain"

    def test_train_test_split_stratified(self, sample_df):
        from sklearn.model_selection import train_test_split
        X = sample_df.drop("target", axis=1)
        y = sample_df["target"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        assert len(X_train) + len(X_test) == len(sample_df)
        assert len(X_train) > len(X_test)

    def test_feature_scaling(self, sample_df):
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        cols   = ["age", "trestbps", "chol"]
        scaled = scaler.fit_transform(sample_df[cols])
        assert abs(scaled.mean()) < 0.5, "Scaled features should be near zero-mean"

    def test_pipeline_builds_correctly(self, sample_df):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer

        num_feats = ["age", "trestbps", "chol", "thalach", "oldpeak"]
        cat_feats = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler())
        ])
        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        preprocessor = ColumnTransformer([
            ("num", numeric_transformer, num_feats),
            ("cat", categorical_transformer, cat_feats)
        ])

        X = sample_df[num_feats + cat_feats]
        result = preprocessor.fit_transform(X)
        assert result.shape[0] == len(sample_df), "Preprocessor output rows should match input"
        assert result.shape[1] > len(num_feats + cat_feats), "OHE should expand categorical features"


# ─────────────────────────────────────────────────────────────────────────────
# Model Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestModelTraining:

    def _build_pipeline(self, clf):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer

        num_feats = ["age", "trestbps", "chol", "thalach", "oldpeak"]
        cat_feats = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

        preprocessor = ColumnTransformer([
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler",  StandardScaler())
            ]), num_feats),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]), cat_feats)
        ])
        return Pipeline([("preprocessor", preprocessor), ("classifier", clf)]), num_feats + cat_feats

    def test_logistic_regression_trains(self, sample_df):
        from sklearn.linear_model import LogisticRegression
        pipeline, feats = self._build_pipeline(LogisticRegression(max_iter=1000, random_state=42))
        X = sample_df[feats]
        y = sample_df["target"]
        pipeline.fit(X, y)
        preds = pipeline.predict(X)
        assert len(preds) == len(y)
        assert set(preds).issubset({0, 1})

    def test_random_forest_trains(self, sample_df):
        from sklearn.ensemble import RandomForestClassifier
        pipeline, feats = self._build_pipeline(RandomForestClassifier(n_estimators=10, random_state=42))
        X = sample_df[feats]
        y = sample_df["target"]
        pipeline.fit(X, y)
        preds  = pipeline.predict(X)
        probas = pipeline.predict_proba(X)
        assert len(preds) == len(y)
        assert probas.shape == (len(y), 2)
        assert np.allclose(probas.sum(axis=1), 1.0)

    def test_model_predict_proba_sums_to_one(self, sample_df):
        from sklearn.ensemble import RandomForestClassifier
        pipeline, feats = self._build_pipeline(RandomForestClassifier(n_estimators=10, random_state=42))
        X = sample_df[feats]
        y = sample_df["target"]
        pipeline.fit(X, y)
        probas = pipeline.predict_proba(X)
        assert np.allclose(probas.sum(axis=1), 1.0, atol=1e-6)

    def test_model_accuracy_above_threshold(self, sample_df):
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split
        pipeline, feats = self._build_pipeline(LogisticRegression(max_iter=1000, random_state=42))
        X = sample_df[feats]
        y = sample_df["target"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        pipeline.fit(X_train, y_train)
        acc = accuracy_score(y_test, pipeline.predict(X_test))
        assert acc >= 0.4, f"Accuracy {acc:.2f} is too low even for random data"

    def test_model_serialization(self, sample_df, tmp_path):
        from sklearn.linear_model import LogisticRegression
        pipeline, feats = self._build_pipeline(LogisticRegression(max_iter=1000, random_state=42))
        X = sample_df[feats]
        y = sample_df["target"]
        pipeline.fit(X, y)

        model_path = tmp_path / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(pipeline, f)

        with open(model_path, "rb") as f:
            loaded = pickle.load(f)

        original_preds = pipeline.predict(X)
        loaded_preds   = loaded.predict(X)
        assert np.array_equal(original_preds, loaded_preds), "Loaded model predictions differ"


# ─────────────────────────────────────────────────────────────────────────────
# API Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAPISchema:

    def test_patient_data_valid(self, sample_patient_dict):
        """Test that valid patient data passes validation."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
        try:
            from app import PatientData
            patient = PatientData(**sample_patient_dict)
            assert patient.age == 63.0
            assert patient.sex == 1.0
        except ImportError:
            pytest.skip("FastAPI app not importable in test environment")

    def test_patient_data_all_features_present(self, sample_patient_dict):
        required = ["age", "sex", "cp", "trestbps", "chol", "fbs",
                    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
        for feat in required:
            assert feat in sample_patient_dict, f"Missing feature in sample: {feat}"

    def test_feature_config_valid(self, tmp_path):
        config = {
            "best_model":           "random_forest",
            "best_roc_auc":         0.92,
            "numerical_features":   ["age", "trestbps", "chol", "thalach", "oldpeak"],
            "categorical_features": ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"],
            "all_features":         ["age", "trestbps", "chol", "thalach", "oldpeak",
                                     "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"],
            "target": "target"
        }
        config_path = tmp_path / "feature_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f)

        with open(config_path) as f:
            loaded = json.load(f)

        assert loaded["best_model"]  == "random_forest"
        assert loaded["best_roc_auc"] == 0.92
        assert len(loaded["all_features"]) == 13
