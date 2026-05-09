# Heart Disease Prediction — MLOps Pipeline
## BITS Pilani | MLOps (S2-25_AMLCSZG523) | Assignment I

**GitHub Repository:** https://github.com/jyotichughgit/heart-disease-mlops  
**Deployed API:** http://34.31.48.169:8000  
**Swagger UI:** http://34.31.48.169:8000/docs

---

## Table of Contents

### Main Report
1. [Setup & Install Instructions](#1-setup--install-instructions)
2. [EDA and Modelling Choices](#2-eda-and-modelling-choices)
3. [Experiment Tracking Summary](#3-experiment-tracking-summary)
4. [Architecture Diagram](#4-architecture-diagram)
5. [CI/CD and Deployment Workflow](#5-cicd-and-deployment-workflow)
6. [Monitoring & Logging](#6-monitoring--logging)

### Appendix
- [A. Project Structure](#a-project-structure)
- [B. GCP Account Setup](#b-gcp-account-setup)
- [C. GKE Cluster Creation](#c-gke-cluster-creation)
- [D. GitHub Secrets Configuration](#d-github-secrets-configuration)
- [E. Docker Installation on Ubuntu](#e-docker-installation-on-ubuntu)
- [F. API Reference](#f-api-reference)
- [G. Dataset Reference](#g-dataset-reference)
- [H. Troubleshooting](#h-troubleshooting)

---

## 1. Setup & Install Instructions

### Prerequisites
- Python 3.10+ (tested on Python 3.12)
- Git 2.x+
- Docker 29.4.3+
- `gcloud` CLI (for GCP deployment)

### Quick Start

```bash
# 1. Clone repository
git clone https://github.com/jyotichughgit/heart-disease-mlops.git
cd heart-disease-mlops

# 2. Create virtual environment (pip)
python3 -m venv venv
source venv/bin/activate      # Linux/Mac
# venv\Scripts\activate       # Windows

# OR using Conda
conda env create -f environment.yml
conda activate heart-disease-mlops

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download dataset (UCI ML Repository)
python scripts/download_data.py
# Expected: Final shape: (303, 14) | Target: {0: 164, 1: 139}
# Missing values filled: ca (mode=0.0), thal (mode=3.0)

# 5. Run EDA
python scripts/eda.py
# Saves 6 plots to screenshots/

# 6. Train model
python src/train.py --data data/heart.csv --output src/models/
# Expected: Best model: logistic_regression (ROC-AUC=0.9665)

# 7. Run hyperparameter tuning
python scripts/tune_and_evaluate.py
# Saves comparison charts to screenshots/

# 8. View MLflow experiment tracking
mlflow ui --backend-store-uri file:./mlruns --port 5000
# Open: http://localhost:5000/#/experiments/161597278421242986/runs

# 9. Test inference and reproducibility
python scripts/inference.py
# Expected: Reproducibility check PASSED — same result across 5 runs

# 10. Start API locally
cd src && uvicorn app:app --reload --port 8000

# 11. Test API
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,
       "fbs":1,"restecg":0,"thalach":150,"exang":0,
       "oldpeak":2.3,"slope":0,"ca":0,"thal":1}'

# 12. Run unit tests
cd .. && pytest tests/ -v --cov=src --cov-fail-under=70
# Expected: 38 passed, coverage 80.16%
```

### Known Issues & Fixes

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: pkg_resources` | `pip install setuptools` |
| `mlflow` import error on Python 3.12 | `pip install --upgrade mlflow` |
| `isort` conflicts with `black` | `setup.cfg` already configures isort with black profile |
| UCI download returns 502 | Script retries 3 times with 2 fallback URLs automatically |
| MLflow shows Traces tab | Go to `http://localhost:5000/#/experiments/161597278421242986/runs` |

---

## 2. EDA and Modelling Choices

### 2.1 Dataset

- **Source:** UCI ML Repository — https://archive.ics.uci.edu/dataset/45/heart+disease
- **Shape:** 303 rows × 14 features
- **Target:** 0=No Disease (164 patients), 1=Disease (139 patients)
- **Class balance:** 54.1% vs 45.9% — well balanced, no oversampling needed

### 2.2 Data Acquisition & Cleaning

```bash
python scripts/download_data.py
```

- Downloads official UCI zip → extracts `processed.cleveland.data`
- Reads with `na_values='?'` → `?` becomes NaN
- **Numerical missing** → filled with **median**
- **Categorical missing** → filled with **mode**
  - `ca`: 4 missing → filled with mode=0.0
  - `thal`: 2 missing → filled with mode=3.0
- `pd.to_numeric(errors='coerce')` + `dropna()` as safety net
- Binarizes target: 0=no disease, 1-4=disease → 1
- Saves to `data/heart.csv`

### 2.3 EDA Visualizations

```bash
python scripts/eda.py
```

| Plot | Description |
|------|-------------|
| `eda_01_class_balance.png` | Bar + pie chart of class distribution |
| `eda_02_feature_histograms.png` | Histograms overlaid by target class |
| `eda_03_correlation_heatmap.png` | Feature correlation heatmap |
| `eda_04_boxplots_by_target.png` | Feature distribution by target class |
| `eda_05_missing_values.png` | Before vs after cleaning comparison |
| `eda_06_categorical_vs_target.png` | Categorical features vs heart disease |

**Key EDA findings:**
- `thalach` (max heart rate) shows strong negative correlation with disease
- `cp` type 0 (asymptomatic) strongly associated with heart disease
- `ca` (major vessels) and `oldpeak` show clear class separation

### 2.4 Feature Engineering

Identical `ColumnTransformer` pipeline used in `train.py`, notebook and `app.py`:

| Type | Features | Preprocessing |
|------|----------|---------------|
| Numerical (5) | age, trestbps, chol, thalach, oldpeak | `SimpleImputer(median)` + `StandardScaler` |
| Categorical (8) | sex, cp, fbs, restecg, exang, slope, ca, thal | `SimpleImputer(mode)` + `OneHotEncoder` |

### 2.5 Model Results

```bash
python src/train.py --data data/heart.csv --output src/models/
python scripts/tune_and_evaluate.py  # GridSearchCV tuning
```

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | CV-AUC |
|-------|----------|-----------|--------|----|---------|--------|
| **Logistic Regression** | **0.8852** | **0.8387** | **0.9286** | **0.8814** | **0.9665** 🏆 | **0.9025** |
| Random Forest | 0.8689 | 0.8125 | 0.9286 | 0.8667 | 0.9416 | 0.8920 |
| Gradient Boosting | 0.8525 | 0.7879 | 0.9286 | 0.8525 | 0.9437 | 0.8281 |

**Best model: Logistic Regression** (ROC-AUC=0.9665)

**Hyperparameters used:**

| Model | Hyperparameters |
|-------|----------------|
| Logistic Regression | C=1.0, max_iter=1000 |
| Random Forest | n_estimators=200, max_depth=8 |
| Gradient Boosting | n_estimators=150, learning_rate=0.1, max_depth=4 |

**Model selection rationale:**
- Highest ROC-AUC on test (0.9665) and CV (0.9025)
- Best precision-recall balance — minimises false negatives critical for medical diagnosis
- CV-AUC closely matches test AUC — no overfitting
- Fast inference — critical for production API

### 2.6 Model Packaging & Reproducibility

```bash
# Saved as pickle (preprocessor + classifier as single pipeline)
src/models/best_model.pkl
src/models/feature_config.json

# Also saved via MLflow
mlflow.sklearn.log_model(pipeline, name)

# Verify reproducibility
python scripts/inference.py
# Output: Reproducibility check PASSED — same result across 5 runs
# Pipeline steps: ColumnTransformer + LogisticRegression
```

---

## 3. Experiment Tracking Summary

**Tool:** MLflow  
**Experiment:** `Heart-Disease-Classification`  
**Runs:** 3 (one per model)

### What is logged per run

| Category | Items |
|----------|-------|
| Parameters | model_type, test_size, random_state, cv_folds, feature lists |
| Metrics | accuracy, precision, recall, f1, roc_auc, cv_roc_auc_mean, cv_roc_auc_std |
| Artifacts | confusion_matrix.png, roc_curve.png, feature_importance.png |
| Models | Full sklearn Pipeline via `mlflow.sklearn.log_model` |

### View MLflow UI

```bash
mlflow ui --backend-store-uri file:./mlruns --port 5000
```

Navigate to: `http://localhost:5000/#/experiments/161597278421242986/runs`

Click **Columns** → add `accuracy`, `precision`, `recall`, `f1`, `roc_auc` to compare all runs.

### Screenshots for Task 3

```
screenshots/task3_mlflow_experiments_list.png  ← experiments page
screenshots/task3_mlflow_runs_table.png        ← 3 runs with metrics
screenshots/task3_mlflow_run_detail.png        ← parameters + metrics
screenshots/task3_mlflow_artifacts.png         ← plots artifacts
```

---

## 4. Architecture Diagram

```
Developer
    │
    │ git push to main
    ▼
┌─────────────────────────────────────────┐
│         GitHub Actions CI/CD            │
│  permissions: id-token: write           │
│                                         │
│  Job 1: Lint                            │
│    └─ flake8 + black + isort            │
│                                         │
│  Job 2: Test                            │
│    └─ 38 pytest tests (coverage 80%)   │
│    └─ uploads test-results artifact     │
│                                         │
│  Job 3: Train                           │
│    └─ download_data.py                  │
│    └─ train.py (3 models + MLflow)      │
│    └─ validate_model.py (AUC >= 0.80)  │
│    └─ uploads model + mlflow artifacts  │
│                                         │
│  Job 4: Build & Push                    │
│    └─ docker build (multi-stage)        │
│    └─ push to Artifact Registry         │
│                                         │
│  Job 5: Deploy                          │
│    └─ gke-gcloud-auth-plugin            │
│    └─ kubectl apply k8s/               │
│    └─ rollout status (300s timeout)     │
│    └─ smoke test /health               │
└───────────────┬─────────────────────────┘
                │ Workload Identity Federation
                │ (no JSON keys — OIDC tokens)
                ▼
┌─────────────────────────────────────────┐
│      Google Artifact Registry           │
│  us-central1-docker.pkg.dev             │
│  heart-disease-repo/heart-disease-api   │
└───────────────┬─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────┐
│     Google Kubernetes Engine (GKE)          │
│     heart-disease-cluster | us-central1-a   │
│     Namespace: mlops                        │
│                                             │
│  ┌─────────────────────────────────────┐   │
│  │         LoadBalancer Service         │   │
│  │      External IP: 34.31.48.169      │   │
│  └──────────────┬──────────────────────┘   │
│                 │                           │
│     ┌───────────┴───────────┐               │
│     ▼                       ▼               │
│  ┌──────────┐         ┌──────────┐         │
│  │ API Pod 1│         │ API Pod 2│         │ ← HPA: 2-6 replicas
│  │ FastAPI  │         │ FastAPI  │         │
│  │ :8000    │         │ :8000    │         │
│  └──────────┘         └──────────┘         │
│                                             │
│  ┌──────────┐         ┌──────────┐         │
│  │Prometheus│◄────────│ Grafana  │         │
│  │  :9090   │         │  :3000   │         │
│  └──────────┘         └──────────┘         │
└─────────────────────────────────────────────┘
```

---

## 5. CI/CD and Deployment Workflow

### 5.1 Pipeline Stages

**Trigger:** Every `git push` to `main` branch  
**File:** `.github/workflows/mlops-pipeline.yml`

| Job | Name | Steps |
|-----|------|-------|
| 1 | Lint | `flake8`, `black --line-length 120`, `isort --profile black --line-length 120` |
| 2 | Unit Tests | 38 pytest tests, coverage 80.16% >= 70%, upload test-results artifact |
| 3 | Train | download data, train 3 models, validate AUC >= 0.80, upload model + mlflow artifacts |
| 4 | Build & Push | multi-stage docker build, push to Artifact Registry with SHA tag |
| 5 | Deploy | gke-gcloud-auth-plugin, kubectl apply, rollout 300s, smoke test /health |

### 5.2 Key Features

- `permissions: id-token: write` — Workload Identity Federation (no JSON keys)
- Actions pinned to Node.js 24 versions (checkout@v4.2.2, setup-python@v5.4.0)
- Model validation gate — AUC >= 0.80 required before Docker build
- Artifacts uploaded per run — test results, model pickle, MLflow runs
- Smoke test after deployment verifies API is responding

### 5.3 Run Locally Before Pushing

```bash
# Lint check
flake8 src/ tests/ --max-line-length=120 --ignore=E501,W503,E203
black --check --line-length 120 src/ tests/
isort --check-only --profile black --line-length 120 src/ tests/

# Auto-fix
black --line-length 120 src/ tests/
isort --profile black --line-length 120 src/ tests/

# Run tests
pytest tests/ -v --cov=src --cov-fail-under=70
```

### 5.4 Docker — Local Test

```bash
bash scripts/docker_build_test.sh
```

Expected:
```
[1/5] Checking model files...        ✓
[2/5] Building Docker image...       ✓
[3/5] Starting container...          ✓
[4/5] Waiting for API to be ready... ✓
[5/5] Testing /predict endpoint...   HTTP 200 ✓
```

### 5.5 Production Deployment — GKE

```bash
# Verify deployment
kubectl get pods -n mlops
kubectl get all -n mlops
kubectl get service heart-disease-api-service -n mlops

# Test endpoints
curl http://34.31.48.169:8000/health

# No Disease prediction
curl -X POST http://34.31.48.169:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,
       "fbs":1,"restecg":0,"thalach":150,"exang":0,
       "oldpeak":2.3,"slope":0,"ca":0,"thal":1}'

# Heart Disease prediction
curl -X POST http://34.31.48.169:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":90,"sex":1,"cp":0,"trestbps":145,"chol":403,
       "fbs":1,"restecg":0,"thalach":150,"exang":1,
       "oldpeak":2.3,"slope":0,"ca":3,"thal":1}'
```

### 5.6 Screenshots for Report

```
Task 5 — CI/CD:
  screenshots/task5_pipeline_overview.png    ← GitHub Actions all jobs green
  screenshots/task5_unit_tests.png           ← 38 tests passed, 80% coverage
  screenshots/task5_lint_pass.png            ← lint job passing

Task 6 — Docker:
  screenshots/task6_docker_build.png         ← docker build output
  screenshots/task6_docker_run.png           ← container running
  screenshots/task6_predict_response.png     ← /predict response

Task 7 — GKE:
  screenshots/task7_gke_pods.png             ← kubectl get pods -n mlops
  screenshots/task7_namespace_mlops.png      ← kubectl get all -n mlops
  screenshots/task7_service_ip.png           ← external IP 34.31.48.169
  screenshots/task7_health_check.png         ← curl /health response
  screenshots/task7_predict_endpoint.png     ← curl /predict response
  screenshots/task7_swagger_ui.png           ← browser /docs
  screenshots/task7_gke_console.png          ← GCP Console Clusters page
  screenshots/task7_gke_workloads.png        ← Workloads → mlops namespace
```

---

## 6. Monitoring & Logging

### 6.1 API Request Logging

Every request logged automatically via middleware in `src/app.py`:

```
2026-05-09 | INFO | POST /predict -> 200 (0.023s)
2026-05-09 | INFO | GET /health -> 200 (0.001s)
2026-05-09 | INFO | Prediction: No Heart Disease | Confidence: 0.8852
```

```bash
# View live logs
kubectl get pods -n mlops -l app=heart-disease-api
kubectl logs -f <pod-name> -n mlops
```

### 6.2 Deploy Prometheus + Grafana

```bash
kubectl apply -f monitoring/prometheus-grafana.yaml
kubectl get pods -n mlops
kubectl get service grafana-service -n mlops
```

### 6.3 Prometheus Metrics

| Metric | Description |
|--------|-------------|
| `api_requests_total` | Total requests by method, endpoint, status |
| `api_request_latency_seconds` | Request latency histogram |
| `predictions_total` | Predictions by label (Heart Disease / No Heart Disease) |

```bash
# View raw metrics
curl http://34.31.48.169:8000/metrics

# Access Prometheus UI
kubectl port-forward -n mlops svc/prometheus-service 9090:9090
# Open: http://localhost:9090
# Query: api_requests_total
```

### 6.4 Grafana Dashboard

```bash
kubectl get service grafana-service -n mlops
# Open: http://<GRAFANA-IP>:3000
# Login: admin / admin123
# Go to: Dashboards → Heart Disease API → Heart Disease API Monitoring
```

Dashboard panels (auto-provisioned — no manual setup):
- Total API Requests
- Total Predictions
- Request Rate per minute
- Request Latency p95
- Predictions by label (pie chart)
- HTTP Status codes over time

### 6.5 GCP Cloud Logging

```
https://console.cloud.google.com/logs?project=heart-disease-mlops-jyotichugh
```
Search filter: `heart-disease-api`

### 6.6 Screenshots for Report

```
screenshots/task8_prometheus_metrics.png  ← curl /metrics output
screenshots/task8_prometheus_ui.png       ← Prometheus UI with api_requests_total query
screenshots/task8_prometheus_graph.png    ← Prometheus graph showing all endpoints
screenshots/task8_grafana_dashboard.png   ← Grafana dashboard with panels
screenshots/task8_api_logs.png            ← kubectl logs showing requests
screenshots/task8_cloud_logging.png       ← GCP Cloud Logging
```

---

---

# Appendix

---

## A. Project Structure

```
heart-disease-mlops/
├── .github/
│   └── workflows/
│       └── mlops-pipeline.yml        # GitHub Actions CI/CD (5 jobs)
├── notebooks/
│   └── 01_eda_and_training.ipynb     # Standalone EDA + training notebook
├── src/
│   ├── app.py                        # FastAPI + Prometheus monitoring
│   ├── train.py                      # Training — 3 models + MLflow + plots
│   └── models/                       # best_model.pkl + feature_config.json (git-ignored)
├── tests/
│   └── test_pipeline.py              # 38 pytest unit tests (coverage 80.16%)
├── docker/
│   └── Dockerfile                    # Multi-stage Docker build
├── k8s/
│   ├── deployment.yaml               # GKE Deployment + LoadBalancer + HPA
│   └── ui-deployment.yaml            # HTML UI deployment
├── monitoring/
│   └── prometheus-grafana.yaml       # Prometheus + Grafana stack
├── scripts/
│   ├── download_data.py              # UCI dataset download + missing value fix
│   ├── eda.py                        # 6 professional EDA plots
│   ├── tune_and_evaluate.py          # GridSearchCV + model comparison
│   ├── validate_model.py             # AUC >= 0.80 gate
│   ├── inference.py                  # Inference + reproducibility proof
│   ├── docker_build_test.sh          # Local Docker build + test
│   └── deploy_ui.sh                  # Deploy HTML UI to GKE
├── ui/
│   └── index.html                    # HTML prediction UI
├── conftest.py                       # Pytest path configuration
├── setup.cfg                         # isort black-compatible profile
├── requirements.txt                  # pip dependencies
├── environment.yml                   # Conda environment
├── .dockerignore                     # Docker build exclusions
├── MLOps_Assignment1_Report.docx     # 10-page assignment report
└── README.md
```

---

## B. GCP Account Setup

### Step 1 — Create Account
1. Go to https://cloud.google.com → sign in with Google account
2. Get **$300 free credits** for 90 days

### Step 2 — Create Project

```bash
gcloud projects create heart-disease-mlops-<your-id> --name="Heart Disease MLOps"
gcloud config set project heart-disease-mlops-<your-id>
export PROJECT_ID=$(gcloud config get-value project)
```

### Step 3 — Enable APIs

```bash
gcloud services enable iam.googleapis.com logging.googleapis.com \
  monitoring.googleapis.com artifactregistry.googleapis.com \
  cloudbuild.googleapis.com container.googleapis.com \
  compute.googleapis.com cloudresourcemanager.googleapis.com
```

### Step 4 — Create Artifact Registry

```bash
gcloud artifacts repositories create heart-disease-repo \
  --repository-format=docker --location=us-central1
```

### Step 5 — Create Service Account

```bash
gcloud iam service-accounts create github-actions-sa \
  --display-name="GitHub Actions Service Account"

for role in roles/container.admin roles/storage.admin \
            roles/iam.serviceAccountUser roles/artifactregistry.admin; do
  gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:github-actions-sa@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="$role"
done
```

### Step 6 — Workload Identity Federation

```bash
# Create pool
gcloud iam workload-identity-pools create "github-pool" \
  --project=$PROJECT_ID --location="global"

# Create provider
gcloud iam workload-identity-pools providers create-oidc "github-provider" \
  --project=$PROJECT_ID --location="global" \
  --workload-identity-pool="github-pool" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
  --attribute-condition="assertion.repository=='jyotichughgit/heart-disease-mlops'" \
  --issuer-uri="https://token.actions.githubusercontent.com"

# Bind service account
export PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format='value(projectNumber)')
gcloud iam service-accounts add-iam-policy-binding \
  github-actions-sa@$PROJECT_ID.iam.gserviceaccount.com \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/github-pool/attribute.repository/jyotichughgit/heart-disease-mlops"
```

---

## C. GKE Cluster Creation

```bash
# Install tools
gcloud components install kubectl gke-gcloud-auth-plugin

# Create cluster
gcloud container clusters create heart-disease-cluster \
  --project=$PROJECT_ID --zone=us-central1-a \
  --machine-type=e2-standard-2 --num-nodes=2 \
  --enable-autoscaling --min-nodes=2 --max-nodes=4

# Connect kubectl
gcloud container clusters get-credentials heart-disease-cluster \
  --zone=us-central1-a --project=$PROJECT_ID

# Grant Artifact Registry access
export PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format='value(projectNumber)')
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$PROJECT_NUMBER-compute@developer.gserviceaccount.com" \
  --role="roles/artifactregistry.reader"

# Fix node service account warning
for role in roles/container.defaultNodeServiceAccount \
            roles/logging.logWriter roles/monitoring.metricWriter \
            roles/monitoring.viewer roles/stackdriver.resourceMetadata.writer; do
  gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$PROJECT_NUMBER-compute@developer.gserviceaccount.com" \
    --role="$role"
done
```

> **Cost:** ~$2-3/day. Delete when not in use:  
> `gcloud container clusters delete heart-disease-cluster --zone=us-central1-a`

---

## D. GitHub Secrets Configuration

Go to: `https://github.com/jyotichughgit/heart-disease-mlops/settings/secrets/actions`

| Secret | Value |
|--------|-------|
| `GCP_PROJECT_ID` | `heart-disease-mlops-jyotichugh` |
| `GCP_PROJECT_NUMBER` | `212262215660` |
| `GCP_SA_EMAIL` | `github-actions-sa@heart-disease-mlops-jyotichugh.iam.gserviceaccount.com` |
| `GCP_WORKLOAD_IDENTITY_PROVIDER` | Output from Workload Identity setup |

---

## E. Docker Installation on Ubuntu

```bash
sudo apt-get remove docker docker-engine docker.io containerd runc
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg lsb-release
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
  sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
sudo usermod -aG docker $USER
newgrp docker
docker --version  # Docker version 29.4.3
docker run hello-world
```

---

## F. API Reference

**Base URL:** `http://34.31.48.169:8000`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Root info |
| `/health` | GET | Health check |
| `/predict` | POST | Single prediction |
| `/predict/batch` | POST | Batch predictions |
| `/model/info` | GET | Model metadata |
| `/metrics` | GET | Prometheus metrics |
| `/docs` | GET | Swagger UI |

**Sample — No Heart Disease:**
```bash
curl -X POST http://34.31.48.169:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,
       "fbs":1,"restecg":0,"thalach":150,"exang":0,
       "oldpeak":2.3,"slope":0,"ca":0,"thal":1}'
```

```json
{
  "prediction": 0,
  "prediction_label": "No Heart Disease",
  "confidence": 0.8852,
  "probability_disease": 0.1148,
  "probability_no_disease": 0.8852,
  "model_name": "logistic_regression"
}
```

**Sample — Heart Disease:**
```bash
curl -X POST http://34.31.48.169:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":90,"sex":1,"cp":0,"trestbps":145,"chol":403,
       "fbs":1,"restecg":0,"thalach":150,"exang":1,
       "oldpeak":2.3,"slope":0,"ca":3,"thal":1}'
```

```json
{
  "prediction": 1,
  "prediction_label": "Heart Disease",
  "confidence": 0.85,
  "probability_disease": 0.85,
  "probability_no_disease": 0.15,
  "model_name": "logistic_regression"
}
```

---

## G. Dataset Reference

| Feature | Description | Type |
|---------|-------------|------|
| age | Age in years | Numerical |
| sex | Sex (1=Male, 0=Female) | Categorical |
| cp | Chest pain type (0-3) | Categorical |
| trestbps | Resting blood pressure (mmHg) | Numerical |
| chol | Serum cholesterol (mg/dl) | Numerical |
| fbs | Fasting blood sugar >120 mg/dl | Categorical |
| restecg | Resting ECG results (0-2) | Categorical |
| thalach | Maximum heart rate achieved | Numerical |
| exang | Exercise induced angina (1=Yes) | Categorical |
| oldpeak | ST depression by exercise | Numerical |
| slope | ST slope (0-2) | Categorical |
| ca | Major vessels coloured (0-3) | Categorical |
| thal | Thalassemia type | Categorical |
| target | Heart disease (1=Yes, 0=No) | Target |

---

## H. Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `HTTP Error 502` on download | UCI site down | Script retries 3× with 2 fallbacks |
| `iam.disableServiceAccountKeyCreation` | GCP org policy | Use Workload Identity Federation |
| `ImagePullBackOff 403` | GKE nodes can't pull image | Grant `roles/artifactregistry.reader` to Compute SA |
| `HPA invalid: AverageUtilization` | Wrong HPA type | Use `Utilization` in k8s/deployment.yaml |
| `gke-gcloud-auth-plugin not found` | Plugin missing | `gcloud components install gke-gcloud-auth-plugin` |
| Rollout timeout | Pods slow to start | Timeout set to 300s in workflow |
| `isort` conflicts with `black` | Style mismatch | `setup.cfg` sets `profile = black` |
| `pkg_resources` not found | Missing on Python 3.12 | `pip install setuptools` |
| MLflow shows Traces tab | MLflow 3.x default | Go to `http://localhost:5000/#/experiments/161597278421242986/runs` |
| GKE node service account warning | Missing IAM roles | Grant `roles/container.defaultNodeServiceAccount` + logging/monitoring roles |
| Grafana shows no data | Prometheus not scraping | Check `kubectl get pods -n mlops` — ensure prometheus pod is Running |
| `id-token` auth error | Missing permissions | Add `permissions: id-token: write` to workflow |
| Node.js 20 deprecation warning | Old action versions | Actions pinned to v4.2.2, v5.4.0 etc. |
