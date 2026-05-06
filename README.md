# Heart Disease Prediction — MLOps Pipeline
## BITS Pilani | MLOps (S2-25_AMLCSZG523) | Assignment I

---

## Table of Contents
1. [Project Structure](#project-structure)
2. [Local Setup](#local-setup)
3. [GCP Account Setup (Step-by-Step)](#gcp-account-setup)
4. [GKE Cluster Creation](#gke-cluster-creation)
5. [GitHub Secrets Configuration](#github-secrets-configuration)
6. [Running the CI/CD Pipeline](#running-the-cicd-pipeline)
7. [Accessing the API](#accessing-the-api)
8. [Monitoring with Prometheus & Grafana](#monitoring)
9. [Manual Docker Testing](#manual-docker-testing)

---

## Project Structure

```
heart-disease-mlops/
├── .github/
│   └── workflows/
│       └── mlops-pipeline.yml        # GitHub Actions CI/CD (Workload Identity)
├── notebooks/
│   └── 01_eda_and_training.ipynb     # EDA + training (Jupyter)
├── src/
│   ├── app.py                        # FastAPI model-serving API
│   ├── train.py                      # Standalone training script
│   └── models/                       # Saved model + feature config (git-ignored)
├── tests/
│   └── test_pipeline.py              # Pytest unit tests (19 tests)
├── docker/
│   └── Dockerfile                    # Multi-stage Docker build
├── k8s/
│   └── deployment.yaml               # GKE Deployment + Service + HPA
├── monitoring/
│   └── prometheus-grafana.yaml       # Prometheus + Grafana on K8s
├── scripts/
│   ├── download_data.py              # Downloads UCI Heart Disease dataset
│   └── validate_model.py             # Model performance gate (AUC threshold)
├── conftest.py                       # Pytest path configuration
├── setup.cfg                         # isort black-compatible profile
├── requirements.txt
└── README.md
```

---

## Local Setup

### Prerequisites
- Python 3.10+ installed
- Git installed
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
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies (includes setuptools for Python 3.12 compatibility)
pip install -r requirements.txt

# 4. Download dataset from official UCI ML Repository
# Source: https://archive.ics.uci.edu/dataset/45/heart+disease
# Downloads heart+disease.zip, extracts processed.cleveland.data,
# adds column headers, handles missing values, saves as data/heart.csv
python scripts/download_data.py
# Expected output: Final shape: (303, 14) | Target: {0: 164, 1: 139}

# 5. Train model (trains 3 models, saves best to src/models/)
python src/train.py --data data/heart.csv --output src/models/
# Expected: Best model: logistic_regression (ROC-AUC=0.9665)

# 6. Start API locally (run from project root)
cd src
uvicorn app:app --reload --port 8000

# 7. Test the API (in a new terminal from project root)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}'

# 8. Run unit tests (from project root)
cd ..
pytest tests/ -v --cov=src
# Expected: 19 passed
```

### Known Issues & Fixes

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: pkg_resources` | `pip install setuptools` |
| `mlflow` version conflict on Python 3.12 | `pip install --upgrade mlflow` |
| `pytest-cov` conflict with mlflow | `pip install pytest-cov --no-deps` then `pip install --upgrade mlflow protobuf` |

---

## GCP Account Setup

### Prerequisites
- Personal Google account at https://cloud.google.com
- `gcloud` CLI installed: https://cloud.google.com/sdk/docs/install

```bash
# Install gcloud CLI on Linux
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Verify installation
gcloud --version

# Initialize and login (opens browser)
gcloud init
```

### Step 1 — Create a GCP Account
1. Go to https://cloud.google.com
2. Click **"Get started for free"** → sign in with Google account
3. Enter billing/card details — GCP gives **$300 free credits** for 90 days
4. You will NOT be charged within free tier limits

### Step 2 — Create a GCP Project

```bash
# Create project
gcloud projects create heart-disease-mlops-<your-id> \
  --name="Heart Disease MLOps"

# Set as active project
gcloud config set project heart-disease-mlops-<your-id>

# Save project ID to environment variable
export PROJECT_ID=$(gcloud config get-value project)
echo $PROJECT_ID
```

### Step 3 — Enable Billing
> Required before enabling APIs. Uses free credits, no charges.

1. Go to https://console.cloud.google.com/billing
2. Create a billing account and link your card
3. Go to https://console.cloud.google.com/billing/projects
4. Find your project → **"Change billing account"** → select your billing account
5. Verify billing is linked:
```bash
gcloud billing projects describe $PROJECT_ID
# Should show: billingEnabled: true
```

### Step 4 — Enable Required APIs

```bash
# Enable APIs one by one (more reliable than all at once)
gcloud services enable iam.googleapis.com
gcloud services enable logging.googleapis.com
gcloud services enable monitoring.googleapis.com
gcloud services enable containerregistry.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable container.googleapis.com
gcloud services enable compute.googleapis.com
```

> **Note:** If `compute.googleapis.com` fails with internal error, go to
> https://console.cloud.google.com/compute in browser, click Enable, accept
> Terms of Service, wait 2-3 minutes, then retry.

### Step 5 — Create Service Account

```bash
# Create service account
gcloud iam service-accounts create github-actions-sa \
  --display-name="GitHub Actions Service Account"

# Grant required roles
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:github-actions-sa@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/container.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:github-actions-sa@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/storage.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:github-actions-sa@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/iam.serviceAccountUser"
```

### Step 6 — Set Up Workload Identity Federation (No Key Files Needed)

> **Why Workload Identity Federation?**
> GCP personal accounts may have `iam.disableServiceAccountKeyCreation` org
> policy enforced by default (Secure by Default enforcement). This prevents
> downloading JSON key files. Workload Identity Federation is Google's
> recommended alternative — more secure and no key management needed.
> GitHub Actions and GCP trust each other via OIDC tokens.

```bash
# 6a. Create Workload Identity Pool
gcloud iam workload-identity-pools create "github-pool" \
  --project=$PROJECT_ID \
  --location="global" \
  --display-name="GitHub Actions Pool"

# 6b. Create OIDC Provider (restricted to your repo only)
gcloud iam workload-identity-pools providers create-oidc "github-provider" \
  --project=$PROJECT_ID \
  --location="global" \
  --workload-identity-pool="github-pool" \
  --display-name="GitHub Provider" \
  --attribute-mapping="google.subject=assertion.sub,attribute.actor=assertion.actor,attribute.repository=assertion.repository" \
  --attribute-condition="assertion.repository=='jyotichughgit/heart-disease-mlops'" \
  --issuer-uri="https://token.actions.githubusercontent.com"

# 6c. Get your project number
export PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format='value(projectNumber)')
echo $PROJECT_NUMBER

# 6d. Bind service account to your GitHub repo
gcloud iam service-accounts add-iam-policy-binding \
  github-actions-sa@$PROJECT_ID.iam.gserviceaccount.com \
  --project=$PROJECT_ID \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/github-pool/attribute.repository/jyotichughgit/heart-disease-mlops"

# 6e. Get provider resource name — COPY THIS OUTPUT for GitHub Secrets
gcloud iam workload-identity-pools providers describe github-provider \
  --project=$PROJECT_ID \
  --location="global" \
  --workload-identity-pool="github-pool" \
  --format="value(name)"
# Example output:
# projects/212262215660/locations/global/workloadIdentityPools/github-pool/providers/github-provider
```

---

## GKE Cluster Creation

```bash
# Create GKE cluster (e2-standard-2 is cost-effective for this assignment)
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

# Get credentials (connects kubectl to your cluster)
gcloud container clusters get-credentials heart-disease-cluster \
  --zone=us-central1-a --project=$PROJECT_ID

# Verify cluster is running
kubectl get nodes
kubectl cluster-info
```

> **Cost tip:** A 2-node e2-standard-2 cluster costs ~$2-3/day.
> Delete when not in use:
> ```bash
> gcloud container clusters delete heart-disease-cluster --zone=us-central1-a
> ```

---

## GitHub Secrets Configuration

Go to your GitHub repo:
**Settings → Secrets and variables → Actions → Repository secrets**

> **Important:** Select **"Repository secrets"** tab, NOT Environment secrets
> or Organization secrets.

Click **"New repository secret"** and add all three:

| Secret Name | Value | How to get it |
|---|---|---|
| `GCP_PROJECT_ID` | `heart-disease-mlops-jyotichugh` | `echo $PROJECT_ID` |
| `GCP_SA_EMAIL` | `github-actions-sa@heart-disease-mlops-jyotichugh.iam.gserviceaccount.com` | Replace with your project ID |
| `GCP_WORKLOAD_IDENTITY_PROVIDER` | `projects/XXXXXX/locations/global/workloadIdentityPools/github-pool/providers/github-provider` | Output from Step 6e above |

After adding all three, verify at:
`https://github.com/jyotichughgit/heart-disease-mlops/settings/secrets/actions`

---

## Running the CI/CD Pipeline

The pipeline triggers automatically on every `git push` to `main`:

```bash
# Make any change and push
git add .
git commit -m "feat: trigger pipeline"
git push origin main
```

### Pipeline Stages (in order)

```
1. Lint       → flake8 + black (line-length 120) + isort (black profile)
2. Test       → pytest 19 unit tests + coverage report
3. Train      → downloads UCI data, trains 3 models, validates AUC >= 0.80
4. Build/Push → builds Docker image, pushes to GCR
5. Deploy     → deploys to GKE, runs smoke test on /health endpoint
```

Monitor at: `https://github.com/jyotichughgit/heart-disease-mlops/actions`

### Linting Rules

```bash
# Run locally before pushing to avoid CI failures
flake8 src/ tests/ --max-line-length=120 --ignore=E501,W503,E203 --exclude=src/models/
black --check --line-length 120 src/ tests/
isort --check-only --diff --profile black --line-length 120 src/ tests/

# Auto-fix formatting
black --line-length 120 src/ tests/
isort --profile black --line-length 120 src/ tests/
```

---

## Accessing the API

After deployment, get the Load Balancer IP:
```bash
kubectl get service heart-disease-api-service -n mlops
# Look for EXTERNAL-IP column
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Root — API info |
| `/health` | GET | Health check |
| `/predict` | POST | Single prediction |
| `/predict/batch` | POST | Batch predictions |
| `/model/info` | GET | Model metadata |
| `/metrics` | GET | Prometheus metrics |
| `/docs` | GET | Swagger UI |

### Sample Prediction Request

```bash
curl -X POST http://<EXTERNAL-IP>:8000/predict \
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
  "timestamp": "2026-05-05T14:22:23.028492"
}
```

---

## Monitoring

### Deploy Prometheus + Grafana to GKE

```bash
kubectl apply -f monitoring/prometheus-grafana.yaml

# Get Grafana external IP
kubectl get service grafana-service -n mlops
```

Open Grafana at `http://<GRAFANA-IP>:3000`
- Login: `admin` / `admin123`
- Add Prometheus data source: `http://prometheus-service:9090`

### Key Metrics to Monitor

| Metric | Description |
|--------|-------------|
| `api_requests_total` | Total requests by endpoint and status |
| `api_request_latency_seconds` | Request latency histogram |
| `predictions_total` | Prediction counts by label |

---

## Manual Docker Testing

```bash
# Build image locally
docker build -f docker/Dockerfile -t heart-disease-api:local .

# Run container
docker run -p 8000:8000 \
  -v $(pwd)/src/models:/app/models \
  heart-disease-api:local

# Test health
curl http://localhost:8000/health

# Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":55,"sex":0,"cp":1,"trestbps":130,"chol":250,"fbs":0,"restecg":1,"thalach":160,"exang":0,"oldpeak":1.0,"slope":2,"ca":0,"thal":2}'
```

---

## MLflow Experiment Tracking

```bash
# Launch MLflow UI locally
mlflow ui --backend-store-uri file:./mlruns --port 5000
# Open: http://localhost:5000
```

### Model Results

| Model | ROC-AUC | CV-AUC |
|-------|---------|--------|
| Logistic Regression | **0.9665** 🏆 | 0.9025 |
| Random Forest | 0.9416 | 0.8920 |
| Gradient Boosting | 0.9437 | 0.8281 |

Best model: **Logistic Regression** saved to `src/models/best_model.pkl`

---

## Architecture

```
Developer → git push → GitHub
                          │
                          ▼
              ┌─────────────────────┐
              │   GitHub Actions     │
              │  1. Lint (flake8,   │
              │     black, isort)   │
              │  2. Test (pytest)   │
              │  3. Train (MLflow)  │
              │  4. Docker Build    │
              │  5. Push → GCR      │
              └────────┬────────────┘
                       │ Workload Identity Federation
                       │ (no JSON keys needed)
                       ▼
              ┌─────────────────────┐
              │  Google Container   │
              │  Registry (GCR)     │
              └────────┬────────────┘
                       │
                       ▼
              ┌──────────────────────────────────┐
              │   Google Kubernetes Engine (GKE)  │
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

---

## Dataset

- **Source:** UCI Machine Learning Repository
- **URL:** https://archive.ics.uci.edu/dataset/45/heart+disease
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
