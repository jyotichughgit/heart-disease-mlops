# Heart Disease Prediction — MLOps Pipeline
## BITS Pilani | MLOps (S2-25_AMLCSZG523) | Assignment I

---

## Table of Contents
1. [Project Structure](#project-structure)
2. [Local Setup](#local-setup)
3. [Task 1 — EDA](#task-1--eda)
4. [Task 2 — Model Training & Tuning](#task-2--model-training--tuning)
5. [Task 3 — Experiment Tracking](#task-3--experiment-tracking)
6. [Task 4 — Model Packaging & Reproducibility](#task-4--model-packaging--reproducibility)
7. [Task 5 — CI/CD Pipeline & Testing](#task-5--cicd-pipeline--testing)
8. [Task 6 — Model Containerization](#task-6--model-containerization)
9. [Task 7 — Production Deployment (GKE)](#task-7--production-deployment-gke)
10. [Task 8 — Monitoring & Logging](#task-8--monitoring--logging)
11. [GCP Account Setup](#gcp-account-setup)
12. [GKE Cluster Creation](#gke-cluster-creation)
13. [GitHub Secrets Configuration](#github-secrets-configuration)
14. [Accessing the API](#accessing-the-api)
15. [Troubleshooting](#troubleshooting)
16. [Dataset](#dataset)

---

## Project Structure

```
heart-disease-mlops/
├── .github/
│   └── workflows/
│       └── mlops-pipeline.yml        # GitHub Actions CI/CD (5 jobs: lint, test, train, build, deploy)
├── notebooks/
│   └── 01_eda_and_training.ipynb     # EDA + MLflow training (Jupyter)
├── src/
│   ├── app.py                        # FastAPI model-serving API with Prometheus monitoring
│   ├── train.py                      # Training script — 3 models, MLflow tracking, plots
│   └── models/                       # Saved model + feature config (git-ignored)
├── tests/
│   └── test_pipeline.py              # Pytest unit tests (29 tests covering all tasks)
├── docker/
│   └── Dockerfile                    # Multi-stage Docker build
├── k8s/
│   ├── deployment.yaml               # GKE Deployment + LoadBalancer Service + HPA
│   └── ui-deployment.yaml            # HTML UI deployment on GKE
├── monitoring/
│   └── prometheus-grafana.yaml       # Prometheus + Grafana on Kubernetes
├── scripts/
│   ├── download_data.py              # Downloads UCI dataset (retries + fallbacks)
│   ├── eda.py                        # EDA — saves 6 professional plots to screenshots/
│   ├── tune_and_evaluate.py          # GridSearchCV tuning + model comparison table
│   ├── validate_model.py             # Model performance gate (AUC >= 0.80)
│   ├── inference.py                  # Inference + reproducibility proof
│   └── docker_build_test.sh          # Local Docker build + test script
├── ui/
│   └── index.html                    # HTML prediction UI (runs on GKE)
├── conftest.py                       # Pytest path configuration
├── setup.cfg                         # isort black-compatible profile
├── requirements.txt                  # Python dependencies (pip)
├── environment.yml                   # Conda environment file
├── .dockerignore                     # Docker build exclusions
└── README.md
```

---

## Local Setup

### Prerequisites
- Python 3.10+ installed
- Git installed
- Docker installed (see Task 6)
- `gcloud` CLI installed (for GCP steps)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/jyotichughgit/heart-disease-mlops.git
cd heart-disease-mlops

# 2. Create virtual environment
# Mac/Linux:
python3 -m venv venv
source venv/bin/activate
# Windows:
# python -m venv venv
# venv\Scripts\activate

# OR use Conda:
# conda env create -f environment.yml
# conda activate heart-disease-mlops

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download dataset
python scripts/download_data.py
# Expected: Final shape: (303, 14) | Target: {0: 164, 1: 139}

# 5. Run EDA
python scripts/eda.py
# Saves 6 plots to screenshots/eda_*.png

# 6. Train model
python src/train.py --data data/heart.csv --output src/models/
# Expected: Best model: logistic_regression (ROC-AUC=0.9665)

# 7. Run hyperparameter tuning and model comparison
python scripts/tune_and_evaluate.py
# Saves comparison table and charts to screenshots/task2_*.png

# 8. View MLflow experiment tracking
mlflow ui --backend-store-uri file:./mlruns --port 5000
# Open: http://localhost:5000

# 9. Test inference and reproducibility
python scripts/inference.py

# 10. Start API locally
cd src
uvicorn app:app --reload --port 8000

# 11. Test API (in a new terminal)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}'

# 12. Run unit tests
cd ..
pytest tests/ -v --cov=src --cov-fail-under=70
# Expected: 29 passed
```

### Known Issues & Fixes

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: pkg_resources` | `pip install setuptools` |
| `mlflow` import error on Python 3.12 | `pip install --upgrade mlflow` |
| `pytest-cov` conflicts with mlflow | `pip install pytest-cov --no-deps` then `pip install --upgrade mlflow protobuf` |
| `isort` conflicts with `black` | `setup.cfg` already configures isort to use black profile |
| UCI download returns 502 | Script retries 3 times with 2 fallback URLs automatically |

---

## Task 1 — EDA

```bash
# Download dataset from official UCI ML Repository
# Source: https://archive.ics.uci.edu/dataset/45/heart+disease
# Download: https://archive.ics.uci.edu/static/public/45/heart+disease.zip
python scripts/download_data.py
```

```bash
# Run EDA — generates 6 professional visualizations
python scripts/eda.py
```

EDA plots saved to `screenshots/`:

| File | Description |
|------|-------------|
| `eda_01_class_balance.png` | Bar chart + pie chart of class distribution |
| `eda_02_feature_histograms.png` | Histograms of all 13 features |
| `eda_03_correlation_heatmap.png` | Feature correlation heatmap |
| `eda_04_boxplots_by_target.png` | Feature distribution by target class |
| `eda_05_missing_values.png` | Missing values per feature |
| `eda_06_categorical_vs_target.png` | Categorical features vs heart disease |

**Dataset summary:**
- Shape: 303 rows × 14 features
- Target: 0=No Disease (164), 1=Disease (139)
- Missing values: `ca` (4), `thal` (2) — filled with median

---

## Task 2 — Model Training & Tuning

```bash
# Train 3 models with MLflow tracking
python src/train.py --data data/heart.csv --output src/models/
```

```bash
# Run GridSearchCV hyperparameter tuning + model comparison
python scripts/tune_and_evaluate.py
# Saves: screenshots/task2_model_comparison.png
#        screenshots/task2_model_performance_chart.png
#        screenshots/task2_tuning_results.json
```

### Model Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | CV-AUC |
|-------|----------|-----------|--------|----|---------|--------|
| Logistic Regression | 0.8689 | 0.8824 | 0.8824 | 0.8824 | **0.9665** 🏆 | 0.9025 |
| Random Forest | 0.8525 | 0.8529 | 0.8824 | 0.8674 | 0.9416 | 0.8920 |
| Gradient Boosting | 0.8689 | 0.8824 | 0.8529 | 0.8674 | 0.9437 | 0.8281 |

Best model: **Logistic Regression** (ROC-AUC=0.9665) saved to `src/models/best_model.pkl`

**Preprocessing pipeline:**
- Numerical features: `SimpleImputer(median)` → `StandardScaler`
- Categorical features: `SimpleImputer(most_frequent)` → `OneHotEncoder`
- Entire pipeline (preprocessor + classifier) saved as single pickle

---

## Task 3 — Experiment Tracking

```bash
# Launch MLflow UI
mlflow ui --backend-store-uri file:./mlruns --port 5000
# Open: http://localhost:5000
```

### Navigating MLflow UI

1. Click **Experiments** tab at the top
2. Go directly to runs:
   ```
   http://localhost:5000/#/experiments/161597278421242986/runs
   ```
3. Click **Columns** button → add `accuracy`, `precision`, `recall`, `f1`, `roc_auc`
4. Click any run to see parameters, metrics and artifacts

### What is logged per run

| Category | Items logged |
|----------|-------------|
| Parameters | model_type, test_size, random_state, cv_folds, feature lists |
| Metrics | accuracy, precision, recall, f1, roc_auc, cv_roc_auc_mean, cv_roc_auc_std |
| Artifacts | confusion_matrix.png, roc_curve.png, feature_importance.png |
| Models | Full sklearn pipeline saved via `mlflow.sklearn.log_model` |

### Screenshots for Task 3 report

```
screenshots/task3_mlflow_experiments_list.png
screenshots/task3_mlflow_runs_table.png
screenshots/task3_mlflow_run_detail.png
screenshots/task3_mlflow_artifacts.png
```

---

## Task 4 — Model Packaging & Reproducibility

### Saved formats

| Format | Location | How |
|--------|----------|-----|
| Pickle | `src/models/best_model.pkl` | `pickle.dump(pipeline, f)` |
| MLflow | `mlruns/` | `mlflow.sklearn.log_model(pipeline, name)` |

### Reproducibility proof

```bash
# Load saved model and run predictions — proves identical results every time
python scripts/inference.py
```

Expected output:
```
=== Reproducibility Check ===
Reproducibility check PASSED — same result across 5 runs
  Prediction:  No Heart Disease
  Confidence:  0.8146

=== Pipeline Steps ===
  preprocessor: ColumnTransformer
  classifier:   LogisticRegression
```

### Conda environment setup

```bash
# Create environment from environment.yml
conda env create -f environment.yml
conda activate heart-disease-mlops
```

---

## Task 5 — CI/CD Pipeline & Testing

### Run tests locally

```bash
pytest tests/ -v --cov=src --cov-fail-under=70
# Expected: 29 passed, coverage >= 70%
```

### Test classes

| Class | Tests | Covers |
|-------|-------|--------|
| `TestDataLoading` | 7 | Dataset validation |
| `TestLoadData` | 3 | `load_data()` in train.py |
| `TestDataPreprocessing` | 4 | Preprocessing pipeline |
| `TestModelTraining` | 5 | LR + RF training |
| `TestTrainAndEvaluate` | 2 | `train_and_evaluate()` in train.py |
| `TestPlotFunctions` | 2 | MLflow plot artifacts |
| `TestMainFunction` | 2 | `main()` end-to-end |
| `TestAPISchema` | 3 | FastAPI schema validation |
| `TestInference` | 3 | Prediction + reproducibility |
| `TestDownloadData` | 7 | Data download validation |

### Linting — run locally before pushing

```bash
# Check
flake8 src/ tests/ --max-line-length=120 --ignore=E501,W503,E203
black --check --line-length 120 src/ tests/
isort --check-only --diff --profile black --line-length 120 src/ tests/

# Auto-fix
black --line-length 120 src/ tests/
isort --profile black --line-length 120 src/ tests/
```

### CI/CD Pipeline stages

```
Push to main
    │
    ▼
1. Lint       → flake8 + black + isort
2. Test       → 29 pytest tests + coverage >= 70%
3. Train      → download data, train 3 models, validate AUC >= 0.80
               → uploads model-artifacts + mlflow-artifacts
4. Build      → docker build → push to Artifact Registry
5. Deploy     → kubectl apply → GKE rollout → smoke test
```

Monitor at: `https://github.com/jyotichughgit/heart-disease-mlops/actions`

---

## Task 6 — Model Containerization

### Install Docker on Ubuntu

```bash
# Remove old versions
sudo apt-get remove docker docker-engine docker.io containerd runc

# Install dependencies
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg lsb-release

# Add Docker GPG key
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
  sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Add repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Verify
docker --version
docker run hello-world
```

### Option 1 — Automated build and test (Recommended)

```bash
bash scripts/docker_build_test.sh
```

Expected output:
```
[1/5] Checking model files...        ✓
[2/5] Building Docker image...       ✓
[3/5] Starting container...          ✓
[4/5] Waiting for API to be ready... ✓
[5/5] Testing endpoints...

--- [POST] /predict with sample patient input ---
HTTP Status: 200
{
    "prediction": 0,
    "prediction_label": "No Heart Disease",
    "confidence": 0.8146,
    "probability_disease": 0.1854,
    "probability_no_disease": 0.8146,
    "model_name": "logistic_regression"
}
Prediction test PASSED ✓
```

### Option 2 — Manual steps

```bash
# Build
docker build -f docker/Dockerfile -t heart-disease-api:local .

# Run
docker run -p 8000:8000 \
  -v $(pwd)/src/models:/app/models \
  -e MODEL_PATH=/app/models/best_model.pkl \
  -e CONFIG_PATH=/app/models/feature_config.json \
  heart-disease-api:local

# Test health
curl http://localhost:8000/health

# Test predict with sample input
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63, "sex": 1, "cp": 3, "trestbps": 145,
    "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150,
    "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
  }'
```

### Screenshots for Task 6 report

```
screenshots/task6_docker_build.png      ← docker build output
screenshots/task6_docker_run.png        ← container start + health check
screenshots/task6_predict_response.png  ← /predict response with sample input
```

---

## Task 7 — Production Deployment (GKE)

See [GCP Account Setup](#gcp-account-setup) and [GKE Cluster Creation](#gke-cluster-creation) sections below.

### Deployed API

| Item | Value |
|------|-------|
| External IP | `34.31.48.169` |
| Health URL | `http://34.31.48.169:8000/health` |
| Predict URL | `http://34.31.48.169:8000/predict` |
| Swagger Docs | `http://34.31.48.169:8000/docs` |
| Replicas | 2 pods + HPA (scales to 6) |

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Root info |
| `/health` | GET | Health check |
| `/predict` | POST | Single prediction |
| `/predict/batch` | POST | Batch predictions |
| `/model/info` | GET | Model metadata |
| `/metrics` | GET | Prometheus metrics |
| `/docs` | GET | Swagger UI |

### Sample Prediction

```bash
curl -X POST http://34.31.48.169:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63, "sex": 1, "cp": 3, "trestbps": 145,
    "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150,
    "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
  }'
```

Expected response:
```json
{
  "prediction": 0,
  "prediction_label": "No Heart Disease",
  "confidence": 0.8146,
  "probability_disease": 0.1854,
  "probability_no_disease": 0.8146,
  "model_name": "logistic_regression",
  "timestamp": "2026-05-06T14:17:27"
}
```

### Kubernetes Namespace

All resources are deployed in the `mlops` namespace:

```bash
# View all resources in mlops namespace
kubectl get all -n mlops

# List all namespaces
kubectl get namespaces
```

### Deploy HTML Prediction UI (Optional)

```bash
# Deploy UI to GKE
bash scripts/deploy_ui.sh

# Get UI external IP
kubectl get service heart-disease-ui-service -n mlops
# Open: http://<UI-IP>
```

### Fix GKE Node Service Account Warning

If you see "Node service account missing roles/container.defaultNodeServiceAccount" in GCP Console:

```bash
export PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format='value(projectNumber)')

# Grant all required node roles
gcloud projects add-iam-policy-binding $PROJECT_ID   --member="serviceAccount:$PROJECT_NUMBER-compute@developer.gserviceaccount.com"   --role="roles/container.defaultNodeServiceAccount"

gcloud projects add-iam-policy-binding $PROJECT_ID   --member="serviceAccount:$PROJECT_NUMBER-compute@developer.gserviceaccount.com"   --role="roles/logging.logWriter"

gcloud projects add-iam-policy-binding $PROJECT_ID   --member="serviceAccount:$PROJECT_NUMBER-compute@developer.gserviceaccount.com"   --role="roles/monitoring.metricWriter"

gcloud projects add-iam-policy-binding $PROJECT_ID   --member="serviceAccount:$PROJECT_NUMBER-compute@developer.gserviceaccount.com"   --role="roles/monitoring.viewer"

gcloud projects add-iam-policy-binding $PROJECT_ID   --member="serviceAccount:$PROJECT_NUMBER-compute@developer.gserviceaccount.com"   --role="roles/stackdriver.resourceMetadata.writer"
```

### Screenshots for Task 7 report

```
screenshots/task7_gke_pods.png          ← kubectl get pods -n mlops
screenshots/task7_namespace_mlops.png   ← kubectl get all -n mlops
screenshots/task7_service_ip.png        ← kubectl get service -n mlops
screenshots/task7_health_check.png      ← curl /health response
screenshots/task7_predict_endpoint.png  ← curl /predict response
screenshots/task7_swagger_ui.png        ← browser showing /docs
screenshots/task7_gke_console.png       ← GCP Console → Kubernetes → Clusters page
screenshots/task7_gke_workloads.png     ← GCP Console → Kubernetes → Workloads → filter mlops
screenshots/task7_ui_deployed.png       ← HTML UI running in browser (optional)
```

---

## Task 8 — Monitoring & Logging

### API Request Logging

Every API request is automatically logged in `src/app.py` via middleware:

```python
logger.info(f"{request.method} {request.url.path} -> {response.status_code} ({latency:.3f}s)")
```

View live logs from running pod:

```bash
# Get pod name
kubectl get pods -n mlops -l app=heart-disease-api

# View live logs
kubectl logs -f <pod-name> -n mlops
```

Expected log output:
```
2026-05-07 | INFO | POST /predict -> 200 (0.023s)
2026-05-07 | INFO | GET /health -> 200 (0.001s)
2026-05-07 | INFO | Prediction: No Heart Disease | Confidence: 0.8146
```

### Deploy Prometheus + Grafana

```bash
# Deploy monitoring stack
kubectl apply -f monitoring/prometheus-grafana.yaml

# Verify pods are running
kubectl get pods -n mlops -l app=prometheus
kubectl get pods -n mlops -l app=grafana

# Get Grafana external IP
kubectl get service grafana-service -n mlops
```

### Access Prometheus UI

```bash
# Port forward Prometheus locally
kubectl port-forward -n mlops svc/prometheus-service 9090:9090

# Open in browser
http://localhost:9090

# Test queries:
# api_requests_total
# predictions_total
# api_request_latency_seconds_bucket
```

### Access Grafana Dashboard

1. Get Grafana IP:
```bash
kubectl get service grafana-service -n mlops
```

2. Open browser: `http://<GRAFANA-IP>:3000`
3. Login: `admin` / `admin123`
4. Go to **Dashboards** → **Heart Disease API** → **Heart Disease API Monitoring**
5. Dashboard auto-loads with these panels:
   - Total API Requests
   - Total Predictions
   - Request Rate per minute
   - Request Rate over time by endpoint
   - Request Latency p95
   - Predictions by label (pie chart)
   - HTTP Status codes

> **Note:** Prometheus data source is pre-configured automatically.
> No manual setup needed — dashboard loads automatically on first login.

### Prometheus Metrics

| Metric | Description |
|--------|-------------|
| `api_requests_total` | Total requests by method, endpoint and status |
| `api_request_latency_seconds` | Request latency histogram |
| `predictions_total` | Total predictions by label (Heart Disease / No Heart Disease) |

View raw metrics:
```bash
curl http://34.31.48.169:8000/metrics
```

### GCP Cloud Logging

GCP automatically captures all pod logs. View at:
```
https://console.cloud.google.com/logs?project=heart-disease-mlops-jyotichugh
```
Search filter: `heart-disease-api`

### Screenshots for Task 8 report

```bash
# Screenshot 1 — Raw Prometheus metrics
curl http://34.31.48.169:8000/metrics
```
📸 Save as `screenshots/task8_prometheus_metrics.png`

```bash
# Screenshot 2 — Prometheus UI with query
kubectl port-forward -n mlops svc/prometheus-service 9090:9090
# Open http://localhost:9090 → type api_requests_total → Execute
```
📸 Save as `screenshots/task8_prometheus_ui.png`

```bash
# Screenshot 3 — Grafana dashboard
# Open http://<GRAFANA-IP>:3000 → login → Dashboards → Heart Disease API Monitoring
```
📸 Save as `screenshots/task8_grafana_dashboard.png`

```bash
# Screenshot 4 — API logs from pod
kubectl logs -f <pod-name> -n mlops
# Make a request in another terminal, watch log appear
```
📸 Save as `screenshots/task8_api_logs.png`

```bash
# Screenshot 5 — GCP Cloud Logging
# https://console.cloud.google.com/logs?project=heart-disease-mlops-jyotichugh
```
📸 Save as `screenshots/task8_cloud_logging.png`

---

## GCP Account Setup

### Prerequisites

```bash
# Install gcloud CLI on Linux
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud --version
gcloud init
```

### Step 1 — Create GCP Account
1. Go to https://cloud.google.com
2. Sign in with personal Google account
3. Get **$300 free credits** for 90 days

### Step 2 — Create GCP Project

```bash
gcloud projects create heart-disease-mlops-<your-id> --name="Heart Disease MLOps"
gcloud config set project heart-disease-mlops-<your-id>
export PROJECT_ID=$(gcloud config get-value project)
```

### Step 3 — Enable Billing
1. Go to https://console.cloud.google.com/billing
2. Create billing account → link to project

```bash
gcloud billing projects describe $PROJECT_ID
# Should show: billingEnabled: true
```

### Step 4 — Enable Required APIs

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

### Step 5 — Create Artifact Registry Repository

```bash
gcloud artifacts repositories create heart-disease-repo \
  --repository-format=docker \
  --location=us-central1 \
  --description="Heart Disease API Docker images"

gcloud artifacts repositories list --location=us-central1
```

### Step 6 — Create Service Account

```bash
gcloud iam service-accounts create github-actions-sa \
  --display-name="GitHub Actions Service Account"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:github-actions-sa@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/container.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:github-actions-sa@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/storage.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:github-actions-sa@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/iam.serviceAccountUser"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:github-actions-sa@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/artifactregistry.admin"
```

### Step 7 — Set Up Workload Identity Federation

> **Why?** GCP personal accounts block JSON key creation by default.
> Workload Identity Federation is Google's recommended keyless alternative.

```bash
# 7a. Create pool
gcloud iam workload-identity-pools create "github-pool" \
  --project=$PROJECT_ID \
  --location="global" \
  --display-name="GitHub Actions Pool"

# 7b. Create provider (restricted to your repo)
gcloud iam workload-identity-pools providers create-oidc "github-provider" \
  --project=$PROJECT_ID \
  --location="global" \
  --workload-identity-pool="github-pool" \
  --display-name="GitHub Provider" \
  --attribute-mapping="google.subject=assertion.sub,attribute.actor=assertion.actor,attribute.repository=assertion.repository" \
  --attribute-condition="assertion.repository=='jyotichughgit/heart-disease-mlops'" \
  --issuer-uri="https://token.actions.githubusercontent.com"

# 7c. Get project number
export PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format='value(projectNumber)')

# 7d. Bind service account to GitHub repo
gcloud iam service-accounts add-iam-policy-binding \
  github-actions-sa@$PROJECT_ID.iam.gserviceaccount.com \
  --project=$PROJECT_ID \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/github-pool/attribute.repository/jyotichughgit/heart-disease-mlops"

# 7e. Get provider name — copy this for GitHub Secrets
gcloud iam workload-identity-pools providers describe github-provider \
  --project=$PROJECT_ID \
  --location="global" \
  --workload-identity-pool="github-pool" \
  --format="value(name)"
```

---

## GKE Cluster Creation

### Step 1 — Install Required Tools

```bash
gcloud components install kubectl
gcloud components install gke-gcloud-auth-plugin
kubectl version --client
gke-gcloud-auth-plugin --version
```

### Step 2 — Create Cluster

```bash
gcloud container clusters create heart-disease-cluster \
  --project=$PROJECT_ID \
  --zone=us-central1-a \
  --machine-type=e2-standard-2 \
  --num-nodes=2 \
  --enable-autoscaling \
  --min-nodes=2 \
  --max-nodes=4 \
  --enable-autorepair \
  --enable-autoupgrade \
  --disk-size=50GB
```

### Step 3 — Connect kubectl

```bash
gcloud container clusters get-credentials heart-disease-cluster \
  --zone=us-central1-a --project=$PROJECT_ID
kubectl get nodes
```

### Step 4 — Grant GKE Nodes Artifact Registry Access

> Without this step pods will fail with `ImagePullBackOff 403 Forbidden`

```bash
export PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format='value(projectNumber)')

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$PROJECT_NUMBER-compute@developer.gserviceaccount.com" \
  --role="roles/artifactregistry.reader"
```

> **Cost tip:** ~$2-3/day. Delete when not in use:
> ```bash
> gcloud container clusters delete heart-disease-cluster --zone=us-central1-a
> ```

---

## GitHub Secrets Configuration

Go to: `https://github.com/jyotichughgit/heart-disease-mlops/settings/secrets/actions`

Select **Repository secrets** tab → **New repository secret**

| Secret Name | Value | How to get |
|---|---|---|
| `GCP_PROJECT_ID` | `heart-disease-mlops-jyotichugh` | `echo $PROJECT_ID` |
| `GCP_PROJECT_NUMBER` | `212262215660` | `echo $PROJECT_NUMBER` |
| `GCP_SA_EMAIL` | `github-actions-sa@heart-disease-mlops-jyotichugh.iam.gserviceaccount.com` | Replace with your project ID |
| `GCP_WORKLOAD_IDENTITY_PROVIDER` | `projects/XXXXXX/locations/global/workloadIdentityPools/github-pool/providers/github-provider` | Output from Step 7e |

---

## Accessing the API

```bash
kubectl get service heart-disease-api-service -n mlops
# Look for EXTERNAL-IP column
```

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `HTTP Error 502` on dataset download | UCI website temporarily down | Script retries 3 times with 2 fallback URLs |
| `iam.disableServiceAccountKeyCreation` | GCP org policy blocks JSON keys | Use Workload Identity Federation |
| `ImagePullBackOff 403 Forbidden` | GKE nodes can't pull from Artifact Registry | Grant `roles/artifactregistry.reader` to Compute Engine SA |
| `HPA invalid: AverageUtilization` | Wrong HPA target type | Use `Utilization` in k8s/deployment.yaml |
| `gke-gcloud-auth-plugin not found` | Plugin not installed | `gcloud components install gke-gcloud-auth-plugin` |
| `Cloud Resource Manager API disabled` | API not enabled | `gcloud services enable cloudresourcemanager.googleapis.com` |
| Rollout timeout in CI/CD | Pods taking >300s | Timeout set to 600s in workflow |
| `isort` conflicts with `black` | Import formatting conflict | `setup.cfg` sets `profile = black` |
| `pkg_resources` not found | Missing setuptools on Python 3.12 | `pip install setuptools` |
| MLflow UI shows Traces tab | MLflow 3.x default | Navigate to `http://localhost:5000/#/experiments/161597278421242986/runs` |
| GKE node service account warning | Missing IAM roles | Grant `roles/container.defaultNodeServiceAccount`, `roles/logging.logWriter`, `roles/monitoring.metricWriter` to Compute Engine SA |
| Grafana shows no data | Prometheus not scraping | Check `kubectl get pods -n mlops` — ensure prometheus pod is Running |
| UI shows blank page | ConfigMap not updated | Re-run `bash scripts/deploy_ui.sh` to refresh ConfigMap |

---

## Dataset

- **Source:** UCI Machine Learning Repository
- **Page:** https://archive.ics.uci.edu/dataset/45/heart+disease
- **Download:** https://archive.ics.uci.edu/static/public/45/heart+disease.zip
- **File used:** `processed.cleveland.data` (inside zip)
- **Shape:** 303 rows × 14 features
- **Target:** Binary (0=No Disease, 1=Disease)
- **Missing values:** `ca` (4), `thal` (2) — filled with median

### Features

| Feature | Description |
|---------|-------------|
| age | Age in years |
| sex | Sex (1=Male, 0=Female) |
| cp | Chest pain type (0-3) |
| trestbps | Resting blood pressure (mmHg) |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar >120 mg/dl (1=True) |
| restecg | Resting ECG results (0-2) |
| thalach | Maximum heart rate achieved |
| exang | Exercise induced angina (1=Yes) |
| oldpeak | ST depression induced by exercise |
| slope | Slope of peak exercise ST segment |
| ca | Number of major vessels (0-3) |
| thal | Thalassemia type |
| target | Heart disease present (1) or absent (0) |

---

## Architecture

```
Developer → git push → GitHub
                          │
                          ▼
              ┌───────────────────────────┐
              │      GitHub Actions        │
              │  1. Lint                   │
              │     - flake8               │
              │     - black                │
              │     - isort                │
              │  2. Test (pytest x29)      │
              │     - coverage >= 70%      │
              │  3. Train (MLflow)         │
              │     - 3 models             │
              │     - uploads artifacts    │
              │  4. Docker Build           │
              │  5. Push to AR             │
              │  6. Deploy to GKE          │
              └────────┬──────────────────┘
                       │ Workload Identity Federation
                       │ (no JSON keys needed)
                       ▼
              ┌──────────────────────────┐
              │   Artifact Registry      │
              │   us-central1-docker     │
              │   .pkg.dev               │
              └────────┬─────────────────┘
                       │
                       ▼
              ┌──────────────────────────────────┐
              │   Google Kubernetes Engine (GKE)  │
              │   heart-disease-cluster           │
              │   us-central1-a                   │
              │                                   │
              │  ┌──────────┐  ┌──────────┐      │
              │  │  API Pod │  │  API Pod │      │ ← 2 replicas + HPA
              │  │ FastAPI  │  │ FastAPI  │      │
              │  └──────────┘  └──────────┘      │
              │         │                        │
              │  ┌──────────────────────┐        │
              │  │  Load Balancer       │        │
              │  │  (External IP:8000)  │        │
              │  └──────────────────────┘        │
              │                                   │
              │  ┌──────────┐  ┌──────────┐      │
              │  │Prometheus│  │ Grafana  │      │
              │  └──────────┘  └──────────┘      │
              └──────────────────────────────────┘
```
