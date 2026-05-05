"""
Heart Disease Prediction API
FastAPI application for serving the ML model with monitoring support.
"""

import json
import logging
import os
import pickle
import time
from datetime import datetime

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field
from starlette.responses import Response

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("heart-disease-api")

# Prometheus Metrics
REQUEST_COUNT = Counter("api_requests_total", "Total API requests", ["method", "endpoint", "status"])
REQUEST_LATENCY = Histogram("api_request_latency_seconds", "Request latency", ["endpoint"])
PREDICTION_COUNT = Counter("predictions_total", "Total predictions made", ["prediction"])

# Model Loading
MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model.pkl")
CONFIG_PATH = os.getenv("CONFIG_PATH", "models/feature_config.json")


def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


try:
    model = load_model()
    config = load_config()
    logger.info(f"Model loaded successfully. Best model: {config.get('best_model')}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None
    config = {}

# FastAPI App
app = FastAPI(
    title="Heart Disease Prediction API",
    description="MLOps Assignment - Heart Disease Risk Classifier",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request / Response Schemas
class PatientData(BaseModel):
    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "age": 63,
                "sex": 1,
                "cp": 3,
                "trestbps": 145,
                "chol": 233,
                "fbs": 1,
                "restecg": 0,
                "thalach": 150,
                "exang": 0,
                "oldpeak": 2.3,
                "slope": 0,
                "ca": 0,
                "thal": 1,
            }
        },
    )

    age: float = Field(..., description="Age in years")
    sex: float = Field(..., description="Sex (1=Male, 0=Female)")
    cp: float = Field(..., description="Chest pain type (0-3)")
    trestbps: float = Field(..., description="Resting blood pressure")
    chol: float = Field(..., description="Serum cholesterol mg/dl")
    fbs: float = Field(..., description="Fasting blood sugar >120 mg/dl (1=True)")
    restecg: float = Field(..., description="Resting ECG results (0-2)")
    thalach: float = Field(..., description="Maximum heart rate achieved")
    exang: float = Field(..., description="Exercise induced angina (1=Yes)")
    oldpeak: float = Field(..., description="ST depression induced by exercise")
    slope: float = Field(..., description="Slope of peak exercise ST segment")
    ca: float = Field(..., description="Number of major vessels (0-3)")
    thal: float = Field(..., description="Thalassemia (1=normal, 2=fixed, 3=reversable)")


class PredictionResponse(BaseModel):
    prediction: int
    prediction_label: str
    confidence: float
    probability_disease: float
    probability_no_disease: float
    model_name: str
    timestamp: str


# Middleware: request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    latency = time.time() - start
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code,
    ).inc()
    REQUEST_LATENCY.labels(endpoint=request.url.path).observe(latency)
    logger.info(f"{request.method} {request.url.path} -> {response.status_code} ({latency:.3f}s)")
    return response


# Endpoints
@app.get("/", tags=["Health"])
def root():
    return {
        "message": "Heart Disease Prediction API",
        "status": "healthy",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_name": config.get("best_model", "unknown"),
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(patient: PatientData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        all_features = config.get(
            "all_features",
            [
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
        )

        input_dict = patient.dict()
        input_df = pd.DataFrame([{k: input_dict.get(k, 0) for k in all_features}])

        prediction = int(model.predict(input_df)[0])
        probas = model.predict_proba(input_df)[0]

        prob_disease = float(probas[1])
        prob_no_disease = float(probas[0])
        confidence = max(prob_disease, prob_no_disease)
        label = "Heart Disease" if prediction == 1 else "No Heart Disease"

        PREDICTION_COUNT.labels(prediction=label).inc()
        logger.info(f"Prediction: {label} | Confidence: {confidence:.4f}")

        return PredictionResponse(
            prediction=prediction,
            prediction_label=label,
            confidence=round(confidence, 4),
            probability_disease=round(prob_disease, 4),
            probability_no_disease=round(prob_no_disease, 4),
            model_name=config.get("best_model", "unknown"),
            timestamp=datetime.utcnow().isoformat(),
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(patients: list[PatientData]):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    results = [predict(p) for p in patients]
    return {"predictions": results, "count": len(results)}


@app.get("/metrics", tags=["Monitoring"])
def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/model/info", tags=["Model"])
def model_info():
    return {
        "model_name": config.get("best_model"),
        "roc_auc": config.get("best_roc_auc"),
        "features": config.get("all_features"),
        "numerical": config.get("numerical_features"),
        "categorical": config.get("categorical_features"),
    }


# Entry point
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
