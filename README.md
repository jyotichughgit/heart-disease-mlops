# Heart Disease Prediction — MLOps Pipeline
## BITS Pilani | MLOps (S2-25_AMLCSZG523) | Assignment I

**GitHub Repository:** https://github.com/jyotichughgit/heart-disease-mlops
**Deployed API:** http://34.31.48.169:8000
**Swagger Docs:** http://34.31.48.169:8000/docs

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
- Python 3.10+
- Git
- Docker (see [Appendix E](#e-docker-installation-on-ubuntu))
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

# 5. Train model
python src/train.py --data data/heart.csv --output src/models/
# Expected: Best model: logistic_regression (ROC-AUC=0.9665)

# 6. Start API locally
cd src && uvicorn app:app --reload --port 8000

# 7. Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,
       "restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}'

# 8. Run unit tests
cd .. && pytest tests/ -v --cov=src --cov-fail-under=70
# Expected: 29 passed, coverage >= 70%
```

### Known Issues & Fixes

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: pkg_resources` | `pip install setuptools` |
| `mlflow` import error on Python 3.12 | `pip install --upgrade mlflow` |
| `pytest-cov` conflicts with mlflow | `pip install pytest-cov --no-deps` then `pip install --upgrade mlflow protobuf` |
| `isort` conflicts with `black` | `setup.cfg` already configures isort to use black profile |
| UCI download returns 502 | Script auto-retries 3 times with 2 fallback URLs |

---

## 2. EDA and Modelling Choices

### 2.1 Dataset

- **Source:** UCI Machine Learning Repository
- **URL:** https://archive.ics.uci.edu/dataset/45/heart+disease
- **File:** `processed.cleveland.data` (extracted from zip)
- **Shape:** 303 rows × 14 features
- **Target:** Binary — 0=No Disease (164 patients), 1=Disease (139 patients)
- **Class balance:** 54.1% vs 45.9% — reasonably balanced, no oversampling needed

### 2.2 Data Acquisition

```bash
python scripts/download_data.py
```

- Downloads official UCI zip from `https://archive.ics.uci.edu/static/public/45/heart+disease.zip`
- Extracts `processed.cleveland.data` — the standard Cleveland subset used in research
- Adds column headers (raw file has none)
- Handles `?` missing values → fills with median
- Binarizes target (0=no disease, 1-4=disease → 1)

### 2.3 Missing Values

| Feature | Missing | Treatment |
|---------|---------|-----------|
| `ca` | 4 rows | Filled with median |
| `thal` | 2 rows | Filled with median |
| All others | 0 | No action needed |

### 2.4 EDA Visualizations

```bash
python scripts/eda.py
# Saves 6 plots to screenshots/
```

| Plot | Description |
|------|-------------|
| `eda_01_class_balance.png` | Bar chart + pie chart of class distribution |
| `eda_02_feature_histograms.png` | Histograms of all 13 features |
| `eda_03_correlation_heatmap.png` | Feature correlation heatmap |
| `eda_04_boxplots_by_target.png` | Feature distribution by target class |
| `eda_05_missing_values.png` | Missing values per feature |
| `eda_06_categorical_vs_target.png` | Categorical features vs heart disease |

**Key EDA findings:**
- `thalach` (max heart rate) shows strong negative correlation with disease
- `cp` (chest pain type) and `ca` (major vessels) are most predictive features
- `chol` and `trestbps` show weaker correlation than expected

### 2.5 Feature Engineering

**Preprocessing pipeline** (sklearn `ColumnTransformer`):

| Feature Type | Features | Steps |
|---|---|---|
| Numerical (5) | age, trestbps, chol, thalach, oldpeak | `SimpleImputer(median)` → `StandardScaler` |
| Categorical (8) | sex, cp, fbs, restecg, exang, slope, ca, thal | `SimpleImputer(most_frequent)` → `OneHotEncoder` |

The entire pipeline (preprocessor + classifier) is saved as a **single pickle** ensuring full reproducibility.

### 2.6 Model Selection

Three classifiers trained and compared:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | CV-AUC |
|-------|----------|-----------|--------|----|---------|--------|
| **Logistic Regression** | **0.8689** | **0.8824** | **0.8824** | **0.8824** | **0.9665** 🏆 | **0.9025** |
| Random Forest | 0.8525 | 0.8529 | 0.8824 | 0.8674 | 0.9416 | 0.8920 |
| Gradient Boosting | 0.8689 | 0.8824 | 0.8529 | 0.8674 | 0.9437 | 0.8281 |

**Best model: Logistic Regression** selected by highest ROC-AUC (0.9665)

**Why Logistic Regression wins:**
- Highest ROC-AUC on both test set (0.9665) and cross-validation (0.9025)
- Best precision and recall balance
- Fast inference — important for production API
- Interpretable — can explain feature weights

### 2.7 Hyperparameter Tuning

```bash
python scripts/tune_and_evaluate.py
```

GridSearchCV with 5-fold StratifiedKFold used for all models:

| Model | Parameter Grid |
|-------|---------------|
| Logistic Regression | `C: [0.01, 0.1, 1.0, 10.0]`, `solver: [lbfgs, liblinear]` |
| Random Forest | `n_estimators: [100, 200]`, `max_depth: [6, 8, None]`, `min_samples_split: [2, 5]` |
| Gradient Boosting | `n_estimators: [100, 150]`, `learning_rate: [0.05, 0.1]`, `max_depth: [3, 4]` |

### 2.8 Evaluation Metrics

- **Accuracy** — overall correctness
- **Precision** — of predicted disease, how many are correct
- **Recall** — of actual disease cases, how many detected
- **F1** — harmonic mean of precision and recall
- **ROC-AUC** — model's ability to distinguish classes (primary metric)
- **CV-AUC** — cross-validation AUC (robustness check)

### 2.9 Model Packaging

```bash
# Saved as pickle
src/models/best_model.pkl       # Full sklearn pipeline
src/models/feature_config.json  # Feature names and config

# Also saved via MLflow
mlflow.sklearn.log_model(pipeline, name)

# Test reproducibility
python scripts/inference.py
# Expected: Reproducibility check PASSED — same result across 5 runs
```

---

## 3. Experiment Tracking Summary

**Tool:** MLflow  
**Experiment name:** `Heart-Disease-Classification`  
**Runs:** 3 (one per model)

### 3.1 What is Logged Per Run

| Category | Items |
|----------|-------|
| Parameters | model_type, test_size, random_state, cv_folds, feature lists |
| Metrics | accuracy, precision, recall, f1, roc_auc, cv_roc_auc_mean, cv_roc_auc_std |
| Artifacts | confusion_matrix.png, roc_curve.png, feature_importance.png |
| Models | Full sklearn pipeline via `mlflow.sklearn.log_model` |

### 3.2 View MLflow UI

```bash
mlflow ui --backend-store-uri file:./mlruns --port 5000
# Open: http://localhost:5000
```

Navigate to runs:
```
http://localhost:5000/#/experiments/161597278421242986/runs
```

Click **Columns** → add `accuracy`, `precision`, `recall`, `f1`, `roc_auc` to see all runs side by side.

### 3.3 Screenshots for Report

```
screenshots/task3_mlflow_experiments_list.png  ← experiments page
screenshots/task3_mlflow_runs_table.png        ← all 3 runs with metrics
screenshots/task3_mlflow_run_detail.png        ← single run parameters + metrics
screenshots/task3_mlflow_artifacts.png         ← confusion matrix + ROC curve plots
```

---

## 4. Architecture Diagram

```
Developer
    │
    │ git push to main
    ▼
┌─────────────────────────────────────────┐
│           GitHub Repository              │
│  jyotichughgit/heart-disease-mlops      │
└───────────────┬─────────────────────────┘
                │ triggers
                ▼
┌─────────────────────────────────────────┐
│         GitHub Actions CI/CD            │
│                                         │
│  Job 1: Lint                            │
│    └─ flake8 + black + isort            │
│                                         │
│  Job 2: Test                            │
│    └─ pytest (29 tests, cov >= 70%)     │
│    └─ uploads: test-results artifact    │
│                                         │
│  Job 3: Train                           │
│    └─ download UCI dataset              │
│    └─ train 3 models (MLflow)           │
│    └─ validate AUC >= 0.80             │
│    └─ uploads: model + mlflow artifacts │
│                                         │
│  Job 4: Build & Push                    │
│    └─ docker build (multi-stage)        │
│    └─ push to Artifact Registry         │
│                                         │
│  Job 5: Deploy                          │
│    └─ kubectl apply k8s/               │
│    └─ rollout status (timeout: 600s)    │
│    └─ smoke test /health               │
└───────────────┬─────────────────────────┘
                │ Workload Identity Federation
                │ (keyless GCP auth)
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
│  │ API Pod 1│         │ API Pod 2│         │ ← HPA scales to 6
│  │ FastAPI  │         │ FastAPI  │         │
│  │ :8000    │         │ :8000    │         │
│  └──────────┘         └──────────┘         │
│                                             │
│  ┌──────────┐         ┌──────────┐         │
│  │Prometheus│◄────────│ Grafana  │         │
│  │  :9090   │         │  :3000   │         │
│  └──────────┘         └──────────┘         │
└─────────────────────────────────────────────┘

API Endpoints:
  GET  /health       → health check
  POST /predict      → single prediction (JSON in, prediction out)
  POST /predict/batch→ batch predictions
  GET  /metrics      → Prometheus metrics
  GET  /docs         → Swagger UI
  GET  /model/info   → model metadata
```

---

## 5. CI/CD and Deployment Workflow

### 5.1 CI/CD Pipeline

**Tool:** GitHub Actions  
**Trigger:** Every `git push` to `main` branch  
**File:** `.github/workflows/mlops-pipeline.yml`

```
Push → Lint → Test → Train → Build → Deploy
  ↓       ↓      ↓      ↓       ↓       ↓
fail    fail   fail   fail    fail   smoke
early  early  early  early   early   test
```

### 5.2 Pipeline Stages

**Stage 1 — Lint**
```bash
flake8 src/ tests/ --max-line-length=120 --ignore=E501,W503,E203
black --check --line-length 120 src/ tests/
isort --check-only --profile black --line-length 120 src/ tests/
```

**Stage 2 — Test**
```bash
pytest tests/ -v --cov=src --cov-fail-under=70 \
  --junitxml=reports/test-results.xml
# 29 tests | coverage >= 70%
# Uploads: test-results artifact
```

**Stage 3 — Train**
```bash
python scripts/download_data.py
python src/train.py --data data/heart.csv --output src/models/
python scripts/validate_model.py --threshold 0.80
# Uploads: model-artifacts + mlflow-artifacts
```

**Stage 4 — Build & Push**
```bash
docker build -f docker/Dockerfile \
  -t us-central1-docker.pkg.dev/$PROJECT_ID/heart-disease-repo/heart-disease-api:$SHA .
docker push us-central1-docker.pkg.dev/$PROJECT_ID/heart-disease-repo/heart-disease-api:$SHA
```

**Stage 5 — Deploy**
```bash
gcloud container clusters get-credentials heart-disease-cluster --zone us-central1-a
kubectl apply -f k8s/
kubectl rollout status deployment/heart-disease-api -n mlops --timeout=600s
curl -f http://$LB_IP:8000/health
```

### 5.3 Run Locally Before Pushing

```bash
# Lint check
flake8 src/ tests/ --max-line-length=120 --ignore=E501,W503,E203
black --check --line-length 120 src/ tests/
isort --check-only --profile black --line-length 120 src/ tests/

# Auto-fix formatting
black --line-length 120 src/ tests/
isort --profile black --line-length 120 src/ tests/

# Run tests
pytest tests/ -v --cov=src --cov-fail-under=70
```

### 5.4 Docker Container — Local Test

```bash
# Build and test locally (Task 6)
bash scripts/docker_build_test.sh
```

Expected:
```
[1/5] Checking model files...        ✓
[2/5] Building Docker image...       ✓
[3/5] Starting container...          ✓
[4/5] Waiting for API to be ready... ✓
[5/5] Testing /predict endpoint...   ✓ HTTP 200
```

### 5.5 Production Deployment on GKE

```bash
# Verify deployment
kubectl get pods -n mlops
kubectl get all -n mlops
kubectl get service heart-disease-api-service -n mlops

# Test endpoints
curl http://34.31.48.169:8000/health
curl -X POST http://34.31.48.169:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,
       "restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}'
```

### 5.6 Screenshots for Report

```
Task 5 — CI/CD:
  screenshots/task5_pipeline_overview.png    ← GitHub Actions all jobs green
  screenshots/task5_unit_tests.png           ← pytest 29 passed
  screenshots/task5_lint_pass.png            ← lint job passing
  screenshots/task5_train_job.png            ← train job with MLflow logging

Task 6 — Docker:
  screenshots/task6_docker_build.png         ← docker build output
  screenshots/task6_docker_run.png           ← container running
  screenshots/task6_predict_response.png     ← /predict response with sample input

Task 7 — GKE Deployment:
  screenshots/task7_gke_pods.png             ← kubectl get pods -n mlops
  screenshots/task7_namespace_mlops.png      ← kubectl get all -n mlops
  screenshots/task7_service_ip.png           ← kubectl get service -n mlops
  screenshots/task7_health_check.png         ← curl /health response
  screenshots/task7_predict_endpoint.png     ← curl /predict response
  screenshots/task7_swagger_ui.png           ← browser /docs
  screenshots/task7_gke_console.png          ← GCP Console Clusters page
  screenshots/task7_gke_workloads.png        ← GCP Console Workloads → mlops namespace
```

---

## 6. Monitoring & Logging

### 6.1 API Request Logging

Every request logged automatically via middleware in `src/app.py`:

```
2026-05-07 | INFO | POST /predict -> 200 (0.023s)
2026-05-07 | INFO | GET /health -> 200 (0.001s)
2026-05-07 | INFO | Prediction: No Heart Disease | Confidence: 0.8146
```

View live logs:
```bash
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

Dashboard panels (auto-provisioned):
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
Search: `heart-disease-api`

### 6.6 Screenshots for Report

```bash
# Screenshot 1 — Raw metrics
curl http://34.31.48.169:8000/metrics
```
📸 `screenshots/task8_prometheus_metrics.png`

```bash
# Screenshot 2 — Prometheus UI
# http://localhost:9090 → query: api_requests_total → Execute
```
📸 `screenshots/task8_prometheus_ui.png`

```bash
# Screenshot 3 — Grafana dashboard
# http://<GRAFANA-IP>:3000 → Dashboards → Heart Disease API Monitoring
```
📸 `screenshots/task8_grafana_dashboard.png`

```bash
# Screenshot 4 — API logs
kubectl logs -f <pod-name> -n mlops
```
📸 `screenshots/task8_api_logs.png`

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
│   └── 01_eda_and_training.ipynb     # EDA + MLflow training (Jupyter)
├── src/
│   ├── app.py                        # FastAPI model-serving API
│   ├── train.py                      # Training — 3 models, MLflow, plots
│   └── models/                       # Saved model + config (git-ignored)
├── tests/
│   └── test_pipeline.py              # 29 pytest unit tests
├── docker/
│   └── Dockerfile                    # Multi-stage Docker build
├── k8s/
│   ├── deployment.yaml               # GKE Deployment + Service + HPA
│   └── ui-deployment.yaml            # HTML UI deployment
├── monitoring/
│   └── prometheus-grafana.yaml       # Prometheus + Grafana stack
├── scripts/
│   ├── download_data.py              # UCI dataset download
│   ├── eda.py                        # EDA — 6 professional plots
│   ├── tune_and_evaluate.py          # GridSearchCV + model comparison
│   ├── validate_model.py             # AUC >= 0.80 gate
│   ├── inference.py                  # Inference + reproducibility
│   ├── docker_build_test.sh          # Local Docker build + test
│   └── deploy_ui.sh                  # Deploy UI to GKE
├── ui/
│   └── index.html                    # HTML prediction UI
├── conftest.py                       # Pytest path configuration
├── setup.cfg                         # isort black-compatible profile
├── requirements.txt                  # pip dependencies
├── environment.yml                   # Conda environment
├── .dockerignore                     # Docker build exclusions
└── README.md
```

---

## B. GCP Account Setup

### Step 1 — Create GCP Account
1. Go to https://cloud.google.com → sign in with personal Google account
2. Get **$300 free credits** for 90 days

### Step 2 — Create Project

```bash
gcloud projects create heart-disease-mlops-<your-id> --name="Heart Disease MLOps"
gcloud config set project heart-disease-mlops-<your-id>
export PROJECT_ID=$(gcloud config get-value project)
```

### Step 3 — Enable Billing
1. Go to https://console.cloud.google.com/billing
2. Create billing account → link to project

### Step 4 — Enable APIs

```bash
gcloud services enable iam.googleapis.com
gcloud services enable logging.googleapis.com
gcloud services enable monitoring.googleapis.com
gcloud services enable artifactregistry.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable container.googleapis.com
gcloud services enable compute.googleapis.com
gcloud services enable cloudresourcemanager.googleapis.com
```

### Step 5 — Create Artifact Registry

```bash
gcloud artifacts repositories create heart-disease-repo \
  --repository-format=docker \
  --location=us-central1 \
  --description="Heart Disease API Docker images"
```

### Step 6 — Create Service Account

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

### Step 7 — Workload Identity Federation

```bash
# Create pool
gcloud iam workload-identity-pools create "github-pool" \
  --project=$PROJECT_ID --location="global" \
  --display-name="GitHub Actions Pool"

# Create provider
gcloud iam workload-identity-pools providers create-oidc "github-provider" \
  --project=$PROJECT_ID --location="global" \
  --workload-identity-pool="github-pool" \
  --attribute-mapping="google.subject=assertion.sub,attribute.actor=assertion.actor,attribute.repository=assertion.repository" \
  --attribute-condition="assertion.repository=='jyotichughgit/heart-disease-mlops'" \
  --issuer-uri="https://token.actions.githubusercontent.com"

# Bind service account
export PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format='value(projectNumber)')
gcloud iam service-accounts add-iam-policy-binding \
  github-actions-sa@$PROJECT_ID.iam.gserviceaccount.com \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/github-pool/attribute.repository/jyotichughgit/heart-disease-mlops"

# Get provider name (for GitHub Secrets)
gcloud iam workload-identity-pools providers describe github-provider \
  --project=$PROJECT_ID --location="global" \
  --workload-identity-pool="github-pool" \
  --format="value(name)"
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
  --enable-autoscaling --min-nodes=2 --max-nodes=4 \
  --enable-autorepair --enable-autoupgrade --disk-size=50GB

# Connect kubectl
gcloud container clusters get-credentials heart-disease-cluster \
  --zone=us-central1-a --project=$PROJECT_ID

# Grant Artifact Registry access to nodes
export PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format='value(projectNumber)')
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$PROJECT_NUMBER-compute@developer.gserviceaccount.com" \
  --role="roles/artifactregistry.reader"

# Grant node service account roles (fixes GCP Console warning)
for role in roles/container.defaultNodeServiceAccount \
            roles/logging.logWriter \
            roles/monitoring.metricWriter \
            roles/monitoring.viewer \
            roles/stackdriver.resourceMetadata.writer; do
  gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$PROJECT_NUMBER-compute@developer.gserviceaccount.com" \
    --role="$role"
done
```

> **Cost tip:** ~$2-3/day. Delete when not in use:
> `gcloud container clusters delete heart-disease-cluster --zone=us-central1-a`

---

## D. GitHub Secrets Configuration

Go to: `https://github.com/jyotichughgit/heart-disease-mlops/settings/secrets/actions`

Select **Repository secrets** → **New repository secret**

| Secret | Value |
|--------|-------|
| `GCP_PROJECT_ID` | `heart-disease-mlops-jyotichugh` |
| `GCP_PROJECT_NUMBER` | `212262215660` |
| `GCP_SA_EMAIL` | `github-actions-sa@heart-disease-mlops-jyotichugh.iam.gserviceaccount.com` |
| `GCP_WORKLOAD_IDENTITY_PROVIDER` | Output from Step 7 above |

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
docker --version
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

**Sample request:**
```bash
curl -X POST http://34.31.48.169:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,
       "fbs":1,"restecg":0,"thalach":150,"exang":0,
       "oldpeak":2.3,"slope":0,"ca":0,"thal":1}'
```

**Sample response:**
```json
{
  "prediction": 0,
  "prediction_label": "No Heart Disease",
  "confidence": 0.8146,
  "probability_disease": 0.1854,
  "probability_no_disease": 0.8146,
  "model_name": "logistic_regression",
  "timestamp": "2026-05-07T12:30:54"
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
| `HTTP Error 502` on dataset download | UCI site down | Script retries 3× with 2 fallbacks |
| `iam.disableServiceAccountKeyCreation` | GCP org policy | Use Workload Identity Federation |
| `ImagePullBackOff 403 Forbidden` | GKE nodes can't pull image | Grant `roles/artifactregistry.reader` to Compute Engine SA |
| `HPA invalid: AverageUtilization` | Wrong HPA type | Use `Utilization` in k8s/deployment.yaml |
| `gke-gcloud-auth-plugin not found` | Plugin missing | `gcloud components install gke-gcloud-auth-plugin` |
| `Cloud Resource Manager API disabled` | API not enabled | `gcloud services enable cloudresourcemanager.googleapis.com` |
| Rollout timeout | Pods slow to start | Timeout set to 600s in workflow |
| `isort` conflicts with `black` | Style mismatch | `setup.cfg` sets `profile = black` |
| `pkg_resources` not found | Missing on Python 3.12 | `pip install setuptools` |
| MLflow shows Traces tab | MLflow 3.x default | Go to `http://localhost:5000/#/experiments/161597278421242986/runs` |
| GKE node service account warning | Missing IAM roles | Grant `roles/container.defaultNodeServiceAccount` + logging/monitoring roles |
| Grafana shows no data | Prometheus not scraping | Check `kubectl get pods -n mlops` |
| UI shows blank page | ConfigMap stale | Re-run `bash scripts/deploy_ui.sh` |
